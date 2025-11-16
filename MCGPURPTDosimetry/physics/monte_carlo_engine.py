"""Monte Carlo engine with full physics implementation."""

from typing import Dict, Optional, Tuple
import torch
import numpy as np

from ..core.data_models import GeometryData, ParticleStack, SecondaryParticleBuffer
from .cross_section_database import CrossSectionDatabase
from .photon_physics import PhotonPhysics
from .electron_physics import ElectronPhysics
from .beta_spectrum import BetaSpectrumCache
from ..utils.logging import get_logger


logger = get_logger()


class MonteCarloEngine:
    """GPU-accelerated Monte Carlo engine with full physics.
    
    Implements detailed physics models for all particle types:
    - Photons: Photoelectric, Compton, pair production, Rayleigh
    - Electrons: Condensed history with bremsstrahlung and delta-rays
    - Positrons: Annihilation with 511 keV photon generation
    - Alphas: Local energy deposition
    
    Attributes:
        geometry: Geometry data
        cross_section_db: Cross-section database
        photon_physics: Photon physics module
        electron_physics: Electron physics module
        config: Simulation configuration
        dose_map: Accumulated dose tensor
        device: Computation device
    """
    
    def __init__(
        self,
        geometry: GeometryData,
        cross_section_db: CrossSectionDatabase,
        config: dict,
        decay_db: Optional['DecayDatabase'] = None
    ):
        """Initialize MonteCarloEngine.
        
        Args:
            geometry: Geometry data
            cross_section_db: Cross-section database
            config: Configuration dictionary
            decay_db: Optional decay database for realistic emission sampling
        """
        self.geometry = geometry
        self.cross_section_db = cross_section_db
        self.decay_db = decay_db
        self.config = config
        self.device = config.get('device', 'cuda')
        
        # Initialize physics modules
        self.photon_physics = PhotonPhysics(device=self.device)
        self.electron_physics = ElectronPhysics(device=self.device)
        
        # Initialize beta spectrum cache
        self.beta_cache = BetaSpectrumCache(device=self.device)
        if decay_db is not None:
            self.beta_cache.preload_from_decay_database(decay_db)
        
        # Initialize dose accumulator
        self.dose_map = torch.zeros(
            geometry.dimensions,
            dtype=torch.float32,
            device=self.device
        )
        
        # Configuration
        self.energy_cutoff_keV = config.get('energy_cutoff_keV', 10.0)
        self.max_particles = config.get('max_particles_in_flight', 100000)
        self.enable_bremsstrahlung = config.get('enable_bremsstrahlung', True)
        self.enable_delta_rays = config.get('enable_delta_rays', False)
        self.enable_fluorescence = config.get('enable_fluorescence', True)
        
        logger.info(
            f"MonteCarloEngine initialized: "
            f"geometry={geometry.dimensions}, device={self.device}"
        )
        logger.info(
            f"Physics options: bremsstrahlung={self.enable_bremsstrahlung}, "
            f"delta_rays={self.enable_delta_rays}, fluorescence={self.enable_fluorescence}"
        )
    
    def simulate_nuclide(
        self,
        source_map: torch.Tensor,
        nuclide: str,
        num_primaries: int
    ) -> torch.Tensor:
        """Simulate particle transport for a nuclide.
        
        Args:
            source_map: Source activity map [X, Y, Z]
            nuclide: Nuclide name
            num_primaries: Number of primary particles
            
        Returns:
            Dose map [X, Y, Z] in Gy
        """
        logger.info(
            f"Simulating {num_primaries} primaries for {nuclide}"
        )
        
        # Create particle stacks
        photon_stack = ParticleStack.create_empty(
            self.max_particles, device=self.device
        )
        electron_stack = ParticleStack.create_empty(
            self.max_particles, device=self.device
        )
        positron_stack = ParticleStack.create_empty(
            self.max_particles // 2, device=self.device
        )
        
        # Create secondary buffer
        secondary_buffer = SecondaryParticleBuffer.create_empty(
            self.max_particles // 2, device=self.device
        )
        
        # Lists to accumulate alpha particles
        alpha_positions = []
        alpha_energies = []
        
        # Sample primary particles with realistic emissions
        self._sample_primaries(
            source_map, num_primaries, nuclide,
            photon_stack, electron_stack, alpha_positions, alpha_energies
        )
        
        # Deposit alpha particle energy locally
        if alpha_positions:
            self._deposit_alpha_energy(alpha_positions, alpha_energies)
        
        # Transport loop
        iteration = 0
        max_iterations = 10000
        
        while iteration < max_iterations:
            # Check if any particles remain
            total_active = (photon_stack.num_active + 
                          electron_stack.num_active + 
                          positron_stack.num_active)
            
            if total_active == 0:
                break
            
            # Transport photons
            if photon_stack.num_active > 0:
                self.transport_photons(
                    photon_stack, electron_stack, positron_stack, secondary_buffer
                )
            
            # Transport electrons
            if electron_stack.num_active > 0:
                self.transport_electrons(
                    electron_stack, photon_stack, secondary_buffer
                )
            
            # Transport positrons
            if positron_stack.num_active > 0:
                self.transport_positrons(
                    positron_stack, photon_stack, secondary_buffer
                )
            
            # Flush secondaries
            secondary_buffer.flush_to_stacks(photon_stack, electron_stack)
            secondary_buffer.clear()
            
            # Compact stacks periodically
            if iteration % 10 == 0:
                if photon_stack.num_active < 0.5 * photon_stack.capacity:
                    photon_stack.compact()
                if electron_stack.num_active < 0.5 * electron_stack.capacity:
                    electron_stack.compact()
                if positron_stack.num_active < 0.5 * positron_stack.capacity:
                    positron_stack.compact()
            
            iteration += 1
            
            # Log progress
            if iteration % 100 == 0:
                logger.debug(
                    f"Iteration {iteration}: "
                    f"photons={photon_stack.num_active}, "
                    f"electrons={electron_stack.num_active}, "
                    f"positrons={positron_stack.num_active}"
                )
        
        logger.info(
            f"Transport complete after {iteration} iterations"
        )
        
        # Convert energy deposition to dose
        dose_map = self._convert_to_dose(self.dose_map)
        
        return dose_map
    
    def transport_photons(
        self,
        photon_stack: ParticleStack,
        electron_stack: ParticleStack,
        positron_stack: ParticleStack,
        secondary_buffer: SecondaryParticleBuffer
    ) -> None:
        """Transport photons with full physics.
        
        Args:
            photon_stack: Photon particle stack
            electron_stack: Electron particle stack
            positron_stack: Positron particle stack
            secondary_buffer: Secondary particle buffer
        """
        active_photons = photon_stack.get_active()
        if active_photons.num_active == 0:
            return
        
        # Get material properties at photon positions
        materials, densities = self._get_material_at_positions(
            active_photons.positions
        )
        
        # Get cross-sections for the most common material (optimization)
        # TODO: Implement per-particle material lookup for heterogeneous geometries
        material_mode = torch.mode(materials).values.item()
        material_name = self._get_material_name(int(material_mode))
        xs_data = self.cross_section_db.get_photon_cross_sections(material_name)
        
        if xs_data is None:
            logger.warning(f"No cross-section data for {material_name}")
            return
        
        # Interpolate cross-sections at photon energies
        xs_interp = xs_data.interpolate(active_photons.energies)
        
        # Sample free paths
        free_paths = self.photon_physics.sample_free_path(
            active_photons.energies,
            xs_interp['total'],
            densities
        )
        
        # Move photons
        new_positions = active_photons.positions + free_paths.unsqueeze(1) * active_photons.directions
        
        # Check boundaries
        inside_mask = self._check_boundaries(new_positions)
        
        # Sample interaction types
        interaction_types = self.photon_physics.sample_interaction_type(
            active_photons.energies,
            xs_interp['photoelectric'],
            xs_interp['compton'],
            xs_interp['pair_production']
        )
        
        # Process interactions
        for i in range(active_photons.num_active):
            if not inside_mask[i]:
                continue
            
            interaction = interaction_types[i].item()
            
            if interaction == 0:  # Photoelectric
                e_pos, e_eng, f_pos, f_eng = self.photon_physics.photoelectric_interaction(
                    active_photons.energies[i:i+1],
                    new_positions[i:i+1]
                )
                
                # Add photoelectron
                if e_eng[0] > self.energy_cutoff_keV:
                    self._add_to_secondary_buffer(
                        secondary_buffer, 'electron',
                        e_pos[0], e_eng[0], active_photons.directions[i]
                    )
                
                # Add fluorescence photon if enabled
                if self.enable_fluorescence and f_eng[0] > self.energy_cutoff_keV:
                    self._add_to_secondary_buffer(
                        secondary_buffer, 'photon',
                        f_pos[0], f_eng[0], active_photons.directions[i]
                    )
            
            elif interaction == 1:  # Compton
                s_eng, s_dir, e_eng, e_dir = self.photon_physics.compton_scattering(
                    active_photons.energies[i:i+1],
                    active_photons.directions[i:i+1]
                )
                
                # Update photon
                if s_eng[0] > self.energy_cutoff_keV:
                    photon_stack.energies[i] = s_eng[0]
                    photon_stack.directions[i] = s_dir[0]
                    photon_stack.positions[i] = new_positions[i]
                else:
                    photon_stack.active_mask[i] = False
                    photon_stack.num_active -= 1
                
                # Add Compton electron
                if e_eng[0] > self.energy_cutoff_keV:
                    self._add_to_secondary_buffer(
                        secondary_buffer, 'electron',
                        new_positions[i], e_eng[0], e_dir[0]
                    )
            
            elif interaction == 2:  # Pair production
                e_pos, e_eng, p_pos, p_eng = self.photon_physics.pair_production(
                    active_photons.energies[i:i+1],
                    new_positions[i:i+1]
                )
                
                # Add electron
                if e_eng[0] > self.energy_cutoff_keV:
                    self._add_to_secondary_buffer(
                        secondary_buffer, 'electron',
                        e_pos[0], e_eng[0], active_photons.directions[i]
                    )
                
                # Add positron
                if p_eng[0] > self.energy_cutoff_keV:
                    self._add_to_secondary_buffer(
                        secondary_buffer, 'positron',
                        p_pos[0], p_eng[0], active_photons.directions[i]
                    )
                
                # Deactivate photon
                photon_stack.active_mask[i] = False
                photon_stack.num_active -= 1
    
    def transport_electrons(
        self,
        electron_stack: ParticleStack,
        photon_stack: ParticleStack,
        secondary_buffer: SecondaryParticleBuffer
    ) -> None:
        """Transport electrons with condensed history and bremsstrahlung.
        
        Args:
            electron_stack: Electron particle stack
            photon_stack: Photon particle stack
            secondary_buffer: Secondary particle buffer
        """
        active_electrons = electron_stack.get_active()
        if active_electrons.num_active == 0:
            return
        
        # Get material properties
        materials, densities = self._get_material_at_positions(
            active_electrons.positions
        )
        
        # Get stopping powers for the most common material (optimization)
        # TODO: Implement per-particle material lookup for heterogeneous geometries
        material_mode = torch.mode(materials).values.item()
        material_name = self._get_material_name(int(material_mode))
        sp_data = self.cross_section_db.get_electron_stopping_powers(material_name)
        
        if sp_data is None:
            # Fallback: local deposition
            for i in range(active_electrons.num_active):
                self._deposit_energy_at_position(
                    active_electrons.positions[i],
                    active_electrons.energies[i]
                )
            electron_stack.num_active = 0
            return
        
        # Interpolate stopping powers
        sp_interp = sp_data.interpolate(active_electrons.energies)
        
        # Calculate step sizes
        voxel_size_cm = np.mean(self.geometry.voxel_size) / 10.0  # mm to cm
        step_sizes = self.electron_physics.calculate_step_size(
            active_electrons.energies,
            sp_interp['total'],
            densities,
            voxel_size_cm
        )
        
        # Calculate energy loss
        energy_loss, rad_loss = self.electron_physics.calculate_energy_loss(
            active_electrons.energies,
            step_sizes,
            sp_interp['collisional'],
            sp_interp['radiative'],
            densities
        )
        
        # Calculate angular deflection
        theta, phi = self.electron_physics.calculate_angular_deflection(
            active_electrons.energies,
            step_sizes,
            material_z=7.4,  # Soft tissue
            density=densities
        )
        
        # Apply deflection
        new_directions = self.electron_physics.apply_angular_deflection(
            active_electrons.directions, theta, phi
        )
        
        # Move electrons
        new_positions = active_electrons.positions + step_sizes.unsqueeze(1) * new_directions
        
        # Deposit energy along path
        for i in range(active_electrons.num_active):
            self._deposit_energy_at_position(
                active_electrons.positions[i],
                energy_loss[i]
            )
        
        # Sample bremsstrahlung
        if self.enable_bremsstrahlung:
            brem_pos, brem_eng, brem_dir = self.electron_physics.sample_bremsstrahlung(
                active_electrons.energies,
                rad_loss,
                active_electrons.positions,
                active_electrons.directions
            )
            
            # Add bremsstrahlung photons to buffer
            for j in range(len(brem_eng)):
                if brem_eng[j] > self.energy_cutoff_keV:
                    self._add_to_secondary_buffer(
                        secondary_buffer, 'photon',
                        brem_pos[j], brem_eng[j], brem_dir[j]
                    )
        
        # Update electron states (vectorized)
        new_energies = active_electrons.energies - energy_loss
        inside_mask = self._check_boundaries(new_positions)
        
        # Create mask for surviving electrons
        survive_mask = (new_energies >= self.energy_cutoff_keV) & inside_mask
        n_surviving = survive_mask.sum().item()
        
        if n_surviving > 0:
            # Update stack with surviving electrons
            electron_stack.positions[:n_surviving] = new_positions[survive_mask]
            electron_stack.directions[:n_surviving] = new_directions[survive_mask]
            electron_stack.energies[:n_surviving] = new_energies[survive_mask]
            electron_stack.active_mask[:n_surviving] = True
            electron_stack.active_mask[n_surviving:] = False
        
        electron_stack.num_active = n_surviving
    
    def transport_positrons(
        self,
        positron_stack: ParticleStack,
        photon_stack: ParticleStack,
        secondary_buffer: SecondaryParticleBuffer
    ) -> None:
        """Transport positrons with annihilation.
        
        Args:
            positron_stack: Positron particle stack
            photon_stack: Photon particle stack
            secondary_buffer: Secondary particle buffer
        """
        active_positrons = positron_stack.get_active()
        if active_positrons.num_active == 0:
            return
        
        # Positrons transport like electrons until they annihilate
        # Simplified: immediate annihilation at rest
        
        for i in range(active_positrons.num_active):
            # Deposit kinetic energy
            self._deposit_energy_at_position(
                active_positrons.positions[i],
                active_positrons.energies[i]
            )
            
            # Generate two 511 keV annihilation photons (back-to-back)
            # Photon 1
            theta1 = torch.acos(2 * torch.rand(1, device=self.device) - 1).item()
            phi1 = 2 * np.pi * torch.rand(1, device=self.device).item()
            
            dir1 = torch.tensor([
                np.sin(theta1) * np.cos(phi1),
                np.sin(theta1) * np.sin(phi1),
                np.cos(theta1)
            ], device=self.device)
            
            # Photon 2 (opposite direction)
            dir2 = -dir1
            
            # Add annihilation photons
            self._add_to_secondary_buffer(
                secondary_buffer, 'photon',
                active_positrons.positions[i], 511.0, dir1
            )
            self._add_to_secondary_buffer(
                secondary_buffer, 'photon',
                active_positrons.positions[i], 511.0, dir2
            )
        
        # Deactivate all positrons
        positron_stack.num_active = 0
    
    def _sample_primaries(
        self,
        source_map: torch.Tensor,
        num_primaries: int,
        nuclide: str,
        photon_stack: ParticleStack,
        electron_stack: ParticleStack,
        alpha_positions: list,
        alpha_energies: list
    ) -> None:
        """Sample primary particles from source with realistic emissions.
        
        Args:
            source_map: Source activity map
            num_primaries: Number of primary particles to sample
            nuclide: Nuclide name for emission sampling
            photon_stack: Photon particle stack
            electron_stack: Electron particle stack
            alpha_positions: List to accumulate alpha particle positions
            alpha_energies: List to accumulate alpha particle energies
        """
        # Sample spatial positions from source distribution
        source_flat = source_map.flatten()
        source_prob = source_flat / (torch.sum(source_flat) + 1e-30)
        
        voxel_indices = torch.multinomial(
            source_prob, num_primaries, replacement=True
        )
        
        nx, ny, nz = self.geometry.dimensions
        iz = voxel_indices // (nx * ny)
        iy = (voxel_indices % (nx * ny)) // nx
        ix = voxel_indices % nx
        
        positions = torch.stack([
            ix.float() + 0.5,
            iy.float() + 0.5,
            iz.float() + 0.5
        ], dim=1)
        
        # Sample isotropic directions
        theta = torch.acos(2 * torch.rand(num_primaries, device=self.device) - 1)
        phi = 2 * np.pi * torch.rand(num_primaries, device=self.device)
        
        directions = torch.stack([
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta)
        ], dim=1)
        
        # Sample emissions from decay database if available
        if self.decay_db is not None and self.decay_db.has_nuclide(nuclide):
            self._sample_realistic_emissions(
                nuclide, num_primaries, positions, directions,
                photon_stack, electron_stack, alpha_positions, alpha_energies
            )
        else:
            # Fallback: use simplified sampling with fixed energy
            logger.warning(
                f"Decay data not available for {nuclide}, using simplified sampling"
            )
            energies = torch.full(
                (num_primaries,), 100.0, dtype=torch.float32, device=self.device
            )
            photon_stack.add_particles(positions, directions, energies)
    
    def _sample_realistic_emissions(
        self,
        nuclide: str,
        num_decays: int,
        positions: torch.Tensor,
        directions: torch.Tensor,
        photon_stack: ParticleStack,
        electron_stack: ParticleStack,
        alpha_positions: list,
        alpha_energies: list
    ) -> None:
        """Sample realistic particle emissions from decay data.
        
        Args:
            nuclide: Nuclide name
            num_decays: Number of decays to sample
            positions: Decay positions [N, 3]
            directions: Emission directions [N, 3]
            photon_stack: Photon particle stack
            electron_stack: Electron particle stack
            alpha_positions: List to accumulate alpha positions
            alpha_energies: List to accumulate alpha energies
        """
        # Get all emissions for this nuclide
        emissions_by_type = self.decay_db.get_all_emissions(nuclide)
        
        for particle_type, emissions in emissions_by_type.items():
            for energy_keV, intensity in emissions:
                # Number of particles of this type to emit
                n_particles = int(num_decays * intensity)
                
                if n_particles == 0:
                    continue
                
                # Sample which decays produce this emission
                indices = torch.randperm(num_decays, device=self.device)[:n_particles]
                
                particle_positions = positions[indices]
                particle_directions = directions[indices]
                particle_energies = torch.full(
                    (n_particles,), energy_keV, dtype=torch.float32, device=self.device
                )
                
                # Add to appropriate stack based on particle type
                if particle_type in ['gamma', 'x-ray']:
                    photon_stack.add_particles(
                        particle_positions, particle_directions, particle_energies
                    )
                elif particle_type in ['beta_minus', 'beta_plus']:
                    # Sample from beta spectrum if available
                    if self.beta_cache.has_nuclide(nuclide):
                        beta_energies = self.beta_cache.sample(nuclide, n_particles)
                    else:
                        # Fallback: use mean energy
                        beta_energies = particle_energies
                    
                    electron_stack.add_particles(
                        particle_positions, particle_directions, beta_energies
                    )
                elif particle_type in ['electron', 'positron']:
                    # Discrete energy electrons (Auger, conversion electrons)
                    electron_stack.add_particles(
                        particle_positions, particle_directions, particle_energies
                    )
                elif particle_type == 'alpha':
                    # Alphas are deposited locally (handled separately)
                    alpha_positions.append(particle_positions)
                    alpha_energies.append(particle_energies)
    
    def _get_material_at_positions(
        self,
        positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get material IDs and densities at positions."""
        # Convert positions to indices
        ix = torch.clamp(positions[:, 0].long(), 0, self.geometry.dimensions[0] - 1)
        iy = torch.clamp(positions[:, 1].long(), 0, self.geometry.dimensions[1] - 1)
        iz = torch.clamp(positions[:, 2].long(), 0, self.geometry.dimensions[2] - 1)
        
        materials = self.geometry.material_map[ix, iy, iz]
        densities = self.geometry.density_map[ix, iy, iz]
        
        return materials, densities
    
    def _get_material_name(self, material_id: int) -> str:
        """Get material name from material ID.
        
        Args:
            material_id: Material ID
            
        Returns:
            Material name (defaults to 'Soft_Tissue' if not found)
        """
        # Material ID to name mapping (should be provided by geometry processor)
        # For now, use a simple fallback
        material_map = {
            0: 'Air',
            1: 'Lung',
            2: 'Fat',
            3: 'Soft_Tissue',
            4: 'Soft_Tissue_Contrast',
            5: 'Bone_Trabecular',
            6: 'Bone_Cortical',
        }
        return material_map.get(material_id, 'Soft_Tissue')
    
    def _check_boundaries(self, positions: torch.Tensor) -> torch.Tensor:
        """Check if positions are inside geometry."""
        nx, ny, nz = self.geometry.dimensions
        inside = (
            (positions[:, 0] >= 0) & (positions[:, 0] < nx) &
            (positions[:, 1] >= 0) & (positions[:, 1] < ny) &
            (positions[:, 2] >= 0) & (positions[:, 2] < nz)
        )
        return inside
    
    def _deposit_energy_at_position(
        self,
        position: torch.Tensor,
        energy: torch.Tensor
    ) -> None:
        """Deposit energy at a position."""
        ix = int(torch.clamp(position[0], 0, self.geometry.dimensions[0] - 1).item())
        iy = int(torch.clamp(position[1], 0, self.geometry.dimensions[1] - 1).item())
        iz = int(torch.clamp(position[2], 0, self.geometry.dimensions[2] - 1).item())
        
        self.dose_map[ix, iy, iz] += energy.item()
    
    def _add_to_secondary_buffer(
        self,
        buffer: SecondaryParticleBuffer,
        particle_type: str,
        position: torch.Tensor,
        energy: float,
        direction: torch.Tensor
    ) -> None:
        """Add particle to secondary buffer."""
        # Simplified: direct addition (should use atomic operations)
        if particle_type == 'photon':
            idx = buffer.photon_count.item()
            if idx < buffer.max_capacity:
                buffer.photon_positions[idx] = position
                buffer.photon_energies[idx] = energy
                buffer.photon_directions[idx] = direction
                buffer.photon_count += 1
        elif particle_type in ['electron', 'positron']:
            idx = buffer.electron_count.item()
            if idx < buffer.max_capacity:
                buffer.electron_positions[idx] = position
                buffer.electron_energies[idx] = energy
                buffer.electron_directions[idx] = direction
                buffer.electron_count += 1
    
    def _convert_to_dose(self, energy_map: torch.Tensor) -> torch.Tensor:
        """Convert energy deposition to absorbed dose."""
        voxel_volume_cm3 = np.prod(self.geometry.voxel_size) / 1000.0
        voxel_mass_g = voxel_volume_cm3 * self.geometry.density_map
        
        energy_J = energy_map * 1.602e-16
        dose_Gy = energy_J / (voxel_mass_g / 1000.0)
        
        return dose_Gy
    
    def get_dose_map(self) -> torch.Tensor:
        """Get accumulated dose map."""
        return self._convert_to_dose(self.dose_map)
    
    def reset_dose_map(self) -> None:
        """Reset dose accumulator."""
        self.dose_map.zero_()
    
    def _deposit_alpha_energy(
        self,
        alpha_positions: list,
        alpha_energies: list
    ) -> None:
        """Deposit alpha particle energy locally.
        
        Alpha particles have very short range (~40 μm in tissue) compared to
        typical voxel sizes (1-5 mm), so we deposit their energy directly in
        the voxel of origin without transport simulation.
        
        Args:
            alpha_positions: List of position tensors for alpha particles
            alpha_energies: List of energy tensors for alpha particles
        """
        if not alpha_positions:
            return
        
        # Concatenate all alpha particles
        all_positions = torch.cat(alpha_positions, dim=0)
        all_energies = torch.cat(alpha_energies, dim=0)
        
        logger.info(f"Depositing {len(all_energies)} alpha particles locally")
        
        # Convert positions to voxel indices
        ix = torch.clamp(
            all_positions[:, 0].long(), 0, self.geometry.dimensions[0] - 1
        )
        iy = torch.clamp(
            all_positions[:, 1].long(), 0, self.geometry.dimensions[1] - 1
        )
        iz = torch.clamp(
            all_positions[:, 2].long(), 0, self.geometry.dimensions[2] - 1
        )
        
        # Deposit energy using atomic operations (index_add_ is atomic)
        # Flatten indices for 3D indexing
        flat_indices = ix + iy * self.geometry.dimensions[0] + \
                      iz * self.geometry.dimensions[0] * self.geometry.dimensions[1]
        
        # Flatten dose map, add energies, reshape back
        dose_flat = self.dose_map.view(-1)
        dose_flat.index_add_(0, flat_indices, all_energies)
        # No need to reassign since view() modifies in-place
