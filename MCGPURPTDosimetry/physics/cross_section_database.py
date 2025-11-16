"""Cross-section database loader for photon and electron interaction data."""

import h5py
from pathlib import Path
from typing import Dict, Optional
import torch
import numpy as np

from ..core.data_models import CrossSectionData, StoppingPowerData
from ..utils.logging import get_logger


logger = get_logger()


class CrossSectionDatabase:
    """Manages photon and electron cross-section data from HDF5 database.
    
    Loads interaction cross-sections for photons (photoelectric, Compton,
    pair production) and stopping powers for electrons (collisional, radiative).
    Data is cached on GPU for fast access during simulation.
    
    Attributes:
        database_path: Path to HDF5 cross-section database
        materials: Dictionary of loaded material data
        device: Device for tensor storage ('cuda' or 'cpu')
        photon_cache: Cached photon cross-section data
        electron_cache: Cached electron stopping power data
    """
    
    def __init__(self, database_path: str, device: str = 'cuda'):
        """Initialize CrossSectionDatabase.
        
        Args:
            database_path: Path to HDF5 cross-section database
            device: Device for tensor storage ('cuda' or 'cpu')
        """
        self.database_path = Path(database_path)
        self.device = device
        self.materials: Dict[str, dict] = {}
        self.photon_cache: Dict[str, CrossSectionData] = {}
        self.electron_cache: Dict[str, StoppingPowerData] = {}
        
        if self.database_path.exists():
            self.load_database()
        else:
            logger.warning(f"Cross-section database not found: {database_path}")
    
    def load_database(self) -> None:
        """Load cross-section database from HDF5 file."""
        logger.info(f"Loading cross-section database from {self.database_path}")
        
        try:
            with h5py.File(self.database_path, 'r') as f:
                # List all materials in database
                material_names = list(f.keys())
                logger.info(f"Found {len(material_names)} materials in database")
                
                for material_name in material_names:
                    self.materials[material_name] = {
                        'photons': material_name + '/photons' in f,
                        'electrons': material_name + '/electrons' in f
                    }
            
            # Validate database structure
            self.validate_database()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load cross-section database: {e}")
    
    def validate_database(self) -> None:
        """Validate database structure and energy range."""
        logger.debug("Validating cross-section database...")
        
        required_materials = [
            'Muscle', 'Soft_Tissue', 'Lung', 'Air', 'Fat',
            'Bone_Cortical', 'Bone_Trabecular', 'Bone_Generic',
            'Iodine_Contrast_Mixture'
        ]
        
        missing_materials = []
        for material in required_materials:
            if material not in self.materials:
                missing_materials.append(material)
        
        if missing_materials:
            logger.warning(
                f"Missing required materials: {', '.join(missing_materials)}"
            )
        
        # Check energy range for one material
        if self.materials:
            sample_material = list(self.materials.keys())[0]
            photon_data = self.get_photon_cross_sections(sample_material)
            if photon_data:
                energy_min = photon_data.energy_grid.min().item()
                energy_max = photon_data.energy_grid.max().item()
                logger.debug(
                    f"Energy range: {energy_min:.1f} eV to {energy_max:.1e} eV"
                )
                
                # Check if range covers 10 eV to 10 MeV
                if energy_min > 10 or energy_max < 1e7:
                    logger.warning(
                        f"Energy range may be insufficient: "
                        f"[{energy_min:.1f}, {energy_max:.1e}] eV"
                    )
        
        logger.debug("Database validation complete")
    
    def get_photon_cross_sections(
        self,
        material_name: str
    ) -> Optional[CrossSectionData]:
        """Get photon cross-section data for a material.
        
        Args:
            material_name: Name of material
            
        Returns:
            CrossSectionData object or None if not found
        """
        # Check cache
        if material_name in self.photon_cache:
            return self.photon_cache[material_name]
        
        # Check if material exists
        if material_name not in self.materials:
            logger.warning(f"Material not found: {material_name}")
            return None
        
        # Load from HDF5
        try:
            with h5py.File(self.database_path, 'r') as f:
                photon_group = f[f'{material_name}/photons']
                
                # Load datasets
                energy_grid = torch.from_numpy(
                    np.array(photon_group['energy_grid'])
                ).float().to(self.device)
                
                photoelectric = torch.from_numpy(
                    np.array(photon_group['photoelectric_cross_section'])
                ).float().to(self.device)
                
                compton = torch.from_numpy(
                    np.array(photon_group['compton_cross_section'])
                ).float().to(self.device)
                
                pair_production = torch.from_numpy(
                    np.array(photon_group['pair_production_cross_section'])
                ).float().to(self.device)
                
                total = torch.from_numpy(
                    np.array(photon_group['total_cross_section'])
                ).float().to(self.device)
                
                cross_section_data = CrossSectionData(
                    energy_grid=energy_grid,
                    photoelectric=photoelectric,
                    compton=compton,
                    pair_production=pair_production,
                    total=total
                )
                
                # Cache result
                self.photon_cache[material_name] = cross_section_data
                
                logger.debug(
                    f"Loaded photon cross-sections for {material_name}: "
                    f"{len(energy_grid)} energy points"
                )
                
                return cross_section_data
                
        except KeyError as e:
            logger.error(
                f"Missing photon data for {material_name}: {e}"
            )
            return None
        except Exception as e:
            logger.error(
                f"Failed to load photon data for {material_name}: {e}"
            )
            return None
    
    def get_electron_stopping_powers(
        self,
        material_name: str
    ) -> Optional[StoppingPowerData]:
        """Get electron stopping power data for a material.
        
        Args:
            material_name: Name of material
            
        Returns:
            StoppingPowerData object or None if not found
        """
        # Check cache
        if material_name in self.electron_cache:
            return self.electron_cache[material_name]
        
        # Check if material exists
        if material_name not in self.materials:
            logger.warning(f"Material not found: {material_name}")
            return None
        
        # Load from HDF5
        try:
            with h5py.File(self.database_path, 'r') as f:
                electron_group = f[f'{material_name}/electrons']
                
                # Load datasets
                energy_grid = torch.from_numpy(
                    np.array(electron_group['energy_grid'])
                ).float().to(self.device)
                
                collisional = torch.from_numpy(
                    np.array(electron_group['collisional_stopping_power'])
                ).float().to(self.device)
                
                radiative = torch.from_numpy(
                    np.array(electron_group['radiative_stopping_power'])
                ).float().to(self.device)
                
                total = collisional + radiative
                
                density_effect = torch.from_numpy(
                    np.array(electron_group['density_effect_correction'])
                ).float().to(self.device)
                
                stopping_power_data = StoppingPowerData(
                    energy_grid=energy_grid,
                    collisional=collisional,
                    radiative=radiative,
                    total=total,
                    density_effect=density_effect
                )
                
                # Cache result
                self.electron_cache[material_name] = stopping_power_data
                
                logger.debug(
                    f"Loaded electron stopping powers for {material_name}: "
                    f"{len(energy_grid)} energy points"
                )
                
                return stopping_power_data
                
        except KeyError as e:
            logger.error(
                f"Missing electron data for {material_name}: {e}"
            )
            return None
        except Exception as e:
            logger.error(
                f"Failed to load electron data for {material_name}: {e}"
            )
            return None
    
    def has_material(self, material_name: str) -> bool:
        """Check if material exists in database.
        
        Args:
            material_name: Name of material
            
        Returns:
            True if material exists
        """
        return material_name in self.materials
    
    def list_materials(self) -> list:
        """Get list of all materials in database.
        
        Returns:
            List of material names
        """
        return list(self.materials.keys())
    
    def preload_materials(self, material_names: list) -> None:
        """Preload cross-section data for multiple materials.
        
        Args:
            material_names: List of material names to preload
        """
        logger.info(f"Preloading {len(material_names)} materials...")
        
        for material_name in material_names:
            self.get_photon_cross_sections(material_name)
            self.get_electron_stopping_powers(material_name)
        
        logger.info("Material preloading complete")
    
    def clear_cache(self) -> None:
        """Clear cached cross-section data."""
        self.photon_cache.clear()
        self.electron_cache.clear()
        logger.debug("Cross-section cache cleared")
