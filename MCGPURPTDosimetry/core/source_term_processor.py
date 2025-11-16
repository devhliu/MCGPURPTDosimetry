"""Source term processor for time-integrated activity calculation."""

from typing import Dict, List, Tuple
import torch
import numpy as np

from .data_models import NuclideData, EmissionData
from ..physics.decay_database import DecayDatabase
from ..utils.logging import get_logger


logger = get_logger()


class SourceTermProcessor:
    """Processes activity maps and calculates time-integrated activities.
    
    Calculates 3D time-integrated activity maps for parent and daughter
    nuclides independently. Handles decay chain resolution and emission
    sampling for Monte Carlo simulation.
    
    Attributes:
        decay_database: Decay database instance
        device: Device for tensor operations
    """
    
    def __init__(self, decay_database: DecayDatabase, device: str = 'cuda'):
        """Initialize SourceTermProcessor.
        
        Args:
            decay_database: Decay database instance
            device: Device for tensor operations ('cuda' or 'cpu')
        """
        self.decay_database = decay_database
        self.device = device
        logger.info("SourceTermProcessor initialized")
    
    def calculate_time_integrated_activity(
        self,
        activity_map: torch.Tensor,
        radionuclide: str
    ) -> Dict[str, torch.Tensor]:
        """Calculate time-integrated activity maps for decay chain.
        
        Args:
            activity_map: Activity map in Bq/pixel [X, Y, Z]
            radionuclide: Parent radionuclide name
            
        Returns:
            Dictionary mapping nuclide names to TIA maps [X, Y, Z]
        """
        logger.info(
            f"Calculating time-integrated activity for {radionuclide}"
        )
        
        # Get decay chain
        decay_chain = self.decay_database.get_decay_chain(radionuclide)
        logger.info(f"Decay chain: {' -> '.join(decay_chain)}")
        
        tia_maps = {}
        
        # Parent nuclide: use input activity map directly
        # (assuming it's already time-integrated)
        tia_maps[radionuclide] = activity_map.clone()
        
        total_activity = torch.sum(activity_map).item()
        logger.info(
            f"Parent {radionuclide} TIA: {total_activity:.2e} Bq·s"
        )
        
        # Daughter nuclides: calculate based on parent activity
        # Simplified model: assume secular equilibrium for daughters
        for i, daughter in enumerate(decay_chain[1:], 1):
            daughter_nuclide = self.decay_database.get_nuclide(daughter)
            if daughter_nuclide is None:
                logger.warning(
                    f"Daughter nuclide {daughter} not in database, skipping"
                )
                continue
            
            # In secular equilibrium, daughter activity equals parent activity
            # This is a simplification; full implementation would use Bateman equations
            tia_maps[daughter] = activity_map.clone()
            
            logger.info(
                f"Daughter {daughter} TIA: {total_activity:.2e} Bq·s "
                f"(secular equilibrium)"
            )
        
        return tia_maps
    
    def get_decay_chain(self, parent_nuclide: str) -> List[str]:
        """Get decay chain for a parent nuclide.
        
        Args:
            parent_nuclide: Parent nuclide name
            
        Returns:
            List of nuclide names in decay chain
        """
        return self.decay_database.get_decay_chain(parent_nuclide)
    
    def sample_emissions(
        self,
        nuclide: str,
        num_decays: torch.Tensor,
        voxel_positions: torch.Tensor
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Sample particle emissions from nuclide decays.
        
        Args:
            nuclide: Nuclide name
            num_decays: Number of decays per voxel [X, Y, Z]
            voxel_positions: Voxel center positions [X, Y, Z, 3]
            
        Returns:
            Dictionary mapping particle types to (positions, energies, weights) tuples
        """
        nuclide_data = self.decay_database.get_nuclide(nuclide)
        if nuclide_data is None:
            raise ValueError(f"Nuclide not found: {nuclide}")
        
        # Get all emissions
        emissions_by_type = self.decay_database.get_all_emissions(nuclide)
        
        sampled_particles = {}
        
        for particle_type, emissions in emissions_by_type.items():
            positions_list = []
            energies_list = []
            weights_list = []
            
            # Sample from each emission line
            for energy_keV, intensity in emissions:
                # Number of particles to sample from this emission
                # intensity is emissions per decay
                n_particles_per_voxel = num_decays * intensity
                
                # Sample particles from non-zero voxels
                nonzero_mask = n_particles_per_voxel > 0
                if not torch.any(nonzero_mask):
                    continue
                
                nonzero_voxels = torch.where(nonzero_mask)
                n_particles_array = n_particles_per_voxel[nonzero_mask]
                
                # For each voxel, sample particles
                for idx in range(len(nonzero_voxels[0])):
                    i, j, k = (
                        nonzero_voxels[0][idx].item(),
                        nonzero_voxels[1][idx].item(),
                        nonzero_voxels[2][idx].item()
                    )
                    
                    n_particles = int(n_particles_array[idx].item())
                    if n_particles == 0:
                        continue
                    
                    # Voxel center position
                    voxel_center = voxel_positions[i, j, k]
                    
                    # Sample positions uniformly within voxel
                    # (simplified: use voxel center)
                    positions = voxel_center.unsqueeze(0).repeat(n_particles, 1)
                    
                    # All particles have same energy
                    energies = torch.full(
                        (n_particles,), energy_keV,
                        dtype=torch.float32, device=self.device
                    )
                    
                    # Statistical weights (all 1.0 for now)
                    weights = torch.ones(
                        n_particles, dtype=torch.float32, device=self.device
                    )
                    
                    positions_list.append(positions)
                    energies_list.append(energies)
                    weights_list.append(weights)
            
            # Concatenate all particles of this type
            if positions_list:
                sampled_particles[particle_type] = (
                    torch.cat(positions_list, dim=0),
                    torch.cat(energies_list, dim=0),
                    torch.cat(weights_list, dim=0)
                )
        
        # Log sampling statistics
        for particle_type, (pos, eng, wgt) in sampled_particles.items():
            logger.debug(
                f"Sampled {len(pos)} {particle_type} particles from {nuclide}"
            )
        
        return sampled_particles
    
    def calculate_num_primaries_per_nuclide(
        self,
        tia_maps: Dict[str, torch.Tensor],
        total_primaries: int
    ) -> Dict[str, int]:
        """Calculate number of primaries to simulate for each nuclide.
        
        Distributes primaries proportionally to total activity.
        
        Args:
            tia_maps: Time-integrated activity maps
            total_primaries: Total number of primaries to simulate
            
        Returns:
            Dictionary mapping nuclide names to number of primaries
        """
        # Calculate total activity for each nuclide
        activities = {
            nuclide: torch.sum(tia_map).item()
            for nuclide, tia_map in tia_maps.items()
        }
        
        total_activity = sum(activities.values())
        
        # Distribute primaries proportionally
        primaries_per_nuclide = {}
        remaining_primaries = total_primaries
        
        for nuclide, activity in activities.items():
            if total_activity > 0:
                fraction = activity / total_activity
                n_primaries = int(total_primaries * fraction)
            else:
                n_primaries = 0
            
            primaries_per_nuclide[nuclide] = n_primaries
            remaining_primaries -= n_primaries
        
        # Distribute remaining primaries to nuclide with highest activity
        if remaining_primaries > 0 and activities:
            max_activity_nuclide = max(activities, key=activities.get)
            primaries_per_nuclide[max_activity_nuclide] += remaining_primaries
        
        logger.info("Primary distribution:")
        for nuclide, n_primaries in primaries_per_nuclide.items():
            logger.info(f"  {nuclide}: {n_primaries} primaries")
        
        return primaries_per_nuclide
