"""Decay database loader for nuclide decay data."""

import json
from pathlib import Path
from typing import Dict, List, Optional
import torch

from ..core.data_models import NuclideData, DecayMode, EmissionData
from ..utils.logging import get_logger


logger = get_logger()


class DecayDatabase:
    """Manages nuclide decay data from JSON database.
    
    Loads and validates decay data including half-lives, decay modes,
    branching ratios, and radiation emissions. Supports decay chain
    resolution to identify daughter nuclides.
    
    Attributes:
        database_path: Path to JSON decay database
        nuclides: Dictionary of loaded nuclide data
        decay_chains: Cache of resolved decay chains
    """
    
    def __init__(self, database_path: str):
        """Initialize DecayDatabase.
        
        Args:
            database_path: Path to JSON decay database file
        """
        self.database_path = Path(database_path)
        self.nuclides: Dict[str, NuclideData] = {}
        self.decay_chains: Dict[str, List[str]] = {}
        
        if self.database_path.exists():
            self.load_database()
        else:
            logger.warning(f"Decay database not found: {database_path}")
    
    def load_database(self) -> None:
        """Load decay database from JSON file."""
        logger.info(f"Loading decay database from {self.database_path}")
        
        try:
            with open(self.database_path, 'r') as f:
                data = json.load(f)
            
            # Parse each nuclide
            for nuclide_name, nuclide_data in data.items():
                self.nuclides[nuclide_name] = self._parse_nuclide(
                    nuclide_name, nuclide_data
                )
            
            logger.info(f"Loaded {len(self.nuclides)} nuclides from database")
            
            # Validate database structure
            self.validate_database()
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in decay database: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load decay database: {e}")
    
    def _parse_nuclide(self, name: str, data: dict) -> NuclideData:
        """Parse nuclide data from JSON.
        
        Args:
            name: Nuclide name
            data: Nuclide data dictionary
            
        Returns:
            NuclideData object
        """
        # Parse decay modes
        decay_modes = {}
        daughters = []
        
        for mode_name, mode_data in data.get('decay_modes', {}).items():
            emissions = []
            for em in mode_data.get('emissions', []):
                # Parse emission data with optional max_energy_keV for beta decay
                emission = EmissionData(
                    particle_type=em['type'],
                    energy_keV=float(em['energy_keV']),
                    intensity=float(em['intensity']),
                    max_energy_keV=float(em['max_energy_keV']) if 'max_energy_keV' in em else None
                )
                emissions.append(emission)
            
            decay_modes[mode_name] = DecayMode(
                mode_type=mode_name,
                branching_ratio=float(mode_data.get('branching_ratio', 0.0)),
                emissions=emissions
            )
            
            # Extract daughter nuclide if present
            if 'daughter' in mode_data:
                daughter = mode_data['daughter']
                if daughter and daughter not in daughters:
                    daughters.append(daughter)
        
        return NuclideData(
            name=name,
            half_life_seconds=float(data['half_life_seconds']),
            decay_modes=decay_modes,
            daughters=daughters
        )
    
    def validate_database(self) -> None:
        """Validate database structure and completeness."""
        logger.debug("Validating decay database...")
        
        required_decay_modes = [
            'gamma', 'beta_minus', 'beta_plus', 'alpha',
            'electron_capture', 'internal_conversion'
        ]
        
        for nuclide_name, nuclide in self.nuclides.items():
            # Check half-life is positive
            if nuclide.half_life_seconds <= 0:
                raise ValueError(
                    f"Invalid half-life for {nuclide_name}: "
                    f"{nuclide.half_life_seconds}"
                )
            
            # Check branching ratios sum to ~1.0
            total_branching = sum(
                mode.branching_ratio for mode in nuclide.decay_modes.values()
            )
            if abs(total_branching - 1.0) > 0.01:
                logger.warning(
                    f"Branching ratios for {nuclide_name} sum to "
                    f"{total_branching:.3f} (expected 1.0)"
                )
            
            # Check emissions have valid intensities
            for mode in nuclide.decay_modes.values():
                for emission in mode.emissions:
                    if emission.intensity < 0 or emission.intensity > 1.0:
                        logger.warning(
                            f"Unusual emission intensity for {nuclide_name}: "
                            f"{emission.intensity}"
                        )
        
        logger.debug("Database validation complete")
    
    def get_nuclide(self, nuclide_name: str) -> Optional[NuclideData]:
        """Get nuclide data by name.
        
        Args:
            nuclide_name: Name of nuclide (e.g., 'Lu-177')
            
        Returns:
            NuclideData object or None if not found
        """
        return self.nuclides.get(nuclide_name)
    
    def get_decay_chain(self, parent_nuclide: str, max_depth: int = 10) -> List[str]:
        """Resolve complete decay chain for a parent nuclide.
        
        Args:
            parent_nuclide: Name of parent nuclide
            max_depth: Maximum chain depth to prevent infinite loops
            
        Returns:
            List of nuclide names in decay chain (including parent)
        """
        # Check cache
        if parent_nuclide in self.decay_chains:
            return self.decay_chains[parent_nuclide]
        
        chain = [parent_nuclide]
        visited = {parent_nuclide}
        current_level = [parent_nuclide]
        
        for depth in range(max_depth):
            next_level = []
            
            for nuclide_name in current_level:
                nuclide = self.get_nuclide(nuclide_name)
                if nuclide is None:
                    continue
                
                for daughter in nuclide.daughters:
                    if daughter not in visited:
                        chain.append(daughter)
                        visited.add(daughter)
                        next_level.append(daughter)
            
            if not next_level:
                break
            
            current_level = next_level
        
        # Cache result
        self.decay_chains[parent_nuclide] = chain
        
        logger.debug(
            f"Decay chain for {parent_nuclide}: {' -> '.join(chain)}"
        )
        
        return chain
    
    def get_all_emissions(
        self,
        nuclide_name: str,
        num_decays: int = 1
    ) -> Dict[str, List[tuple]]:
        """Get all emissions from a nuclide's decay.
        
        Args:
            nuclide_name: Name of nuclide
            num_decays: Number of decays to sample
            
        Returns:
            Dictionary mapping particle types to lists of (energy, intensity) tuples
        """
        nuclide = self.get_nuclide(nuclide_name)
        if nuclide is None:
            raise ValueError(f"Nuclide not found: {nuclide_name}")
        
        emissions_by_type = {}
        
        for mode in nuclide.decay_modes.values():
            for emission in mode.emissions:
                particle_type = emission.particle_type
                
                if particle_type not in emissions_by_type:
                    emissions_by_type[particle_type] = []
                
                # Scale intensity by branching ratio
                scaled_intensity = emission.intensity * mode.branching_ratio
                
                emissions_by_type[particle_type].append(
                    (emission.energy_keV, scaled_intensity)
                )
        
        return emissions_by_type
    
    def has_nuclide(self, nuclide_name: str) -> bool:
        """Check if nuclide exists in database.
        
        Args:
            nuclide_name: Name of nuclide
            
        Returns:
            True if nuclide exists
        """
        return nuclide_name in self.nuclides
    
    def list_nuclides(self) -> List[str]:
        """Get list of all nuclides in database.
        
        Returns:
            List of nuclide names
        """
        return list(self.nuclides.keys())
