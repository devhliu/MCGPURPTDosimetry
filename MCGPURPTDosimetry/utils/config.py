"""Configuration management for dosimetry simulations."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import yaml
from pathlib import Path

from ..physics_data import (
    DEFAULT_DECAY_DATABASE,
    DEFAULT_CROSS_SECTION_DATABASE
)


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo dosimetry simulation.
    
    Attributes:
        radionuclide: Name of radionuclide to simulate (e.g., 'Lu-177')
        num_primaries: Number of primary particles to simulate
        energy_cutoff_keV: Energy cutoff threshold in keV for photons and electrons
        num_batches: Number of batches for uncertainty calculation
        hu_to_material_lut: HU-to-material lookup table with multi-range support
        output_format: Output format ('file' or 'object')
        output_path: Path for output files (required if output_format='file')
        random_seed: Random seed for reproducibility (None for random)
        device: Computation device ('cuda' or 'cpu')
        max_particles_in_flight: Maximum particles in transport stacks
        decay_database_path: Path to decay database JSON file
        cross_section_database_path: Path to cross-section database HDF5 file
    """
    radionuclide: str
    num_primaries: int
    energy_cutoff_keV: float = 10.0
    num_batches: int = 10
    hu_to_material_lut: Optional[Dict[str, List[Tuple[float, float]]]] = None
    output_format: str = 'file'
    output_path: Optional[str] = None
    random_seed: Optional[int] = None
    device: str = 'cuda'
    max_particles_in_flight: int = 100000
    decay_database_path: Optional[str] = None
    cross_section_database_path: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Set default database paths if not provided
        if self.decay_database_path is None:
            if DEFAULT_DECAY_DATABASE is None:
                raise ValueError(
                    "No decay database path provided and default database not found. "
                    "Please generate physics databases or provide explicit path."
                )
            self.decay_database_path = DEFAULT_DECAY_DATABASE
        if self.cross_section_database_path is None:
            if DEFAULT_CROSS_SECTION_DATABASE is None:
                raise ValueError(
                    "No cross-section database path provided and default database not found. "
                    "Please generate physics databases or provide explicit path."
                )
            self.cross_section_database_path = DEFAULT_CROSS_SECTION_DATABASE
        
        self._validate()
    
    def _validate(self) -> None:
        """Validate configuration parameters."""
        import torch
        from ..utils.logging import get_logger
        logger = get_logger()
        
        # Validate radionuclide
        if not self.radionuclide or not isinstance(self.radionuclide, str):
            raise ValueError("radionuclide must be a non-empty string")
        
        # Validate num_primaries
        if self.num_primaries <= 0:
            raise ValueError(f"num_primaries must be positive, got {self.num_primaries}")
        
        # Validate energy_cutoff_keV
        if self.energy_cutoff_keV <= 0:
            raise ValueError(f"energy_cutoff_keV must be positive, got {self.energy_cutoff_keV}")
        
        # Validate num_batches
        if self.num_batches < 2:
            raise ValueError(f"num_batches must be at least 2 for uncertainty calculation, got {self.num_batches}")
        
        # Validate output_format
        if self.output_format not in ['file', 'object']:
            raise ValueError(f"output_format must be 'file' or 'object', got {self.output_format}")
        
        # Validate output_path for file output
        if self.output_format == 'file' and not self.output_path:
            raise ValueError("output_path is required when output_format='file'")
        
        # Validate device with automatic fallback
        if self.device not in ['cuda', 'cpu']:
            raise ValueError(f"device must be 'cuda' or 'cpu', got {self.device}")
        
        if self.device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = 'cpu'
        
        # Validate max_particles_in_flight
        if self.max_particles_in_flight <= 0:
            raise ValueError(f"max_particles_in_flight must be positive, got {self.max_particles_in_flight}")
        
        # Set default HU-to-material LUT if not provided
        if self.hu_to_material_lut is None:
            self.hu_to_material_lut = self._get_default_hu_lut()
    
    @staticmethod
    def _get_default_hu_lut() -> Dict[str, List[Tuple[float, float]]]:
        """Get default HU-to-material lookup table with contrast handling."""
        return {
            'Air': [(-1000, -950)],
            'Lung': [(-950, -150)],
            'Fat': [(-150, -50)],
            'Soft_Tissue': [(-50, 100)],
            'Soft_Tissue_Contrast': [(100, 300)],  # Contrast-enhanced soft tissue
            'Bone_Trabecular': [(300, 700)],
            'Bone_Cortical': [(700, 3000)],
        }
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'SimulationConfig':
        """Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            SimulationConfig instance
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert HU LUT from YAML format if present
        if 'hu_to_material_lut' in config_dict:
            lut = {}
            for material, ranges in config_dict['hu_to_material_lut'].items():
                lut[material] = [tuple(r) for r in ranges]
            config_dict['hu_to_material_lut'] = lut
        
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            yaml_path: Path to save YAML configuration
        """
        config_dict = {
            'radionuclide': self.radionuclide,
            'num_primaries': self.num_primaries,
            'energy_cutoff_keV': self.energy_cutoff_keV,
            'num_batches': self.num_batches,
            'hu_to_material_lut': {
                material: [list(r) for r in ranges]
                for material, ranges in self.hu_to_material_lut.items()
            },
            'output_format': self.output_format,
            'output_path': self.output_path,
            'random_seed': self.random_seed,
            'device': self.device,
            'max_particles_in_flight': self.max_particles_in_flight,
            'decay_database_path': self.decay_database_path,
            'cross_section_database_path': self.cross_section_database_path,
        }
        
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @staticmethod
    def get_default_config() -> 'SimulationConfig':
        """Get a default configuration for testing.
        
        Returns:
            SimulationConfig with default values
        """
        return SimulationConfig(
            radionuclide='Lu-177',
            num_primaries=100000,
            energy_cutoff_keV=10.0,
            num_batches=10,
            output_format='file',
            output_path='./results/'
        )
