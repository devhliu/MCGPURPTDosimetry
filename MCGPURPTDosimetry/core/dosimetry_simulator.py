"""Main dosimetry simulator orchestration class."""

from typing import Union, Dict, Optional, List
import nibabel as nib
import torch

from .input_manager import InputManager
from .geometry_processor import GeometryProcessor
from .data_models import GeometryData
from ..utils.config import SimulationConfig
from ..utils.logging import setup_logger, get_logger
from ..utils.validation import validate_config


class DosimetrySimulator:
    """Main orchestration class for GPU-accelerated dosimetry simulation.
    
    This class coordinates all components of the simulation pipeline:
    - Input image loading and validation
    - Geometry processing
    - Source term calculation
    - Monte Carlo transport
    - Dose synthesis and output
    
    Attributes:
        config: Simulation configuration
        input_manager: Input image manager
        geometry_processor: Geometry processor
        logger: Logger instance
    """
    
    def __init__(self, config: SimulationConfig):
        """Initialize DosimetrySimulator.
        
        Args:
            config: Simulation configuration
        """
        # Validate configuration
        validate_config(config)
        self.config = config
        
        # Set up logging
        log_file = None
        if config.output_format == 'file' and config.output_path:
            from pathlib import Path
            log_dir = Path(config.output_path)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = str(log_dir / 'simulation.log')
        
        self.logger = setup_logger(log_file=log_file)
        self.logger.info("DosimetrySimulator initialized")
        self.logger.info(f"Configuration: radionuclide={config.radionuclide}, "
                        f"primaries={config.num_primaries}, device={config.device}")
        
        # Initialize components
        self.input_manager = InputManager()
        self.geometry_processor = GeometryProcessor(
            hu_to_material_lut=config.hu_to_material_lut
        )
        
        # Set random seed if provided
        if config.random_seed is not None:
            torch.manual_seed(config.random_seed)
            self.logger.info(f"Random seed set to {config.random_seed}")
        
        # Reset GPU memory stats for accurate tracking
        if config.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    
    def run(
        self,
        ct_image: Union[str, nib.Nifti1Image],
        activity_map: Union[str, nib.Nifti1Image],
        tissue_masks: Optional[Dict[str, Union[str, nib.Nifti1Image]]] = None,
        mask_priority_order: Optional[List[str]] = None,
        use_ct_density: bool = True
    ) -> Dict:
        """Run complete dosimetry simulation.
        
        Args:
            ct_image: CT image (file path or nibabel object)
            activity_map: Activity map in Bq/pixel (file path or nibabel object)
            tissue_masks: Optional dictionary mapping tissue names to mask sources
            mask_priority_order: Optional priority order for overlapping masks
            use_ct_density: If True, use CT-derived density in masked regions
            
        Returns:
            Dictionary with simulation results
        """
        import time
        from ..physics import DecayDatabase, CrossSectionDatabase
        from ..physics.monte_carlo_engine import MonteCarloEngine
        from .source_term_processor import SourceTermProcessor
        from .dose_synthesis import DoseSynthesis
        
        start_time = time.time()
        
        self.logger.info("=" * 60)
        self.logger.info("Starting dosimetry simulation")
        self.logger.info("=" * 60)
        
        # Step 1: Load and validate input images
        self.logger.info("Step 1: Loading input images")
        ct_tensor = self.input_manager.load_ct_image(ct_image, device=self.config.device)
        activity_tensor = self.input_manager.load_activity_image(activity_map, device=self.config.device)
        self.input_manager.validate_image_compatibility()
        
        # Step 2: Load tissue masks (if provided)
        mask_tensors = None
        if tissue_masks is not None:
            self.logger.info("Step 2a: Loading tissue masks")
            mask_tensors = self.input_manager.load_segmentation_masks(
                tissue_masks, device=self.config.device
            )
        
        # Step 2b: Process geometry
        self.logger.info("Step 2b: Processing geometry")
        geometry = self.geometry_processor.create_geometry_data(
            ct_tensor,
            self.input_manager.get_voxel_dimensions(),
            self.input_manager.get_affine_matrix(),
            mask_dict=mask_tensors,
            mask_priority_order=mask_priority_order,
            use_ct_density=use_ct_density
        )
        
        # Step 3: Load physics databases
        self.logger.info("Step 3: Loading physics databases")
        decay_db = DecayDatabase(self.config.decay_database_path)
        xs_db = CrossSectionDatabase(
            self.config.cross_section_database_path,
            device=self.config.device
        )
        
        # Step 4: Calculate source terms
        self.logger.info("Step 4: Calculating source terms")
        source_processor = SourceTermProcessor(decay_db, device=self.config.device)
        tia_maps = source_processor.calculate_time_integrated_activity(
            activity_tensor,
            self.config.radionuclide
        )
        
        # Calculate primaries per nuclide
        primaries_per_nuclide = source_processor.calculate_num_primaries_per_nuclide(
            tia_maps,
            self.config.num_primaries
        )
        
        # Step 5: Initialize Monte Carlo engine
        self.logger.info("Step 5: Initializing Monte Carlo engine")
        mc_config = {
            'device': self.config.device,
            'energy_cutoff_keV': self.config.energy_cutoff_keV,
            'max_particles_in_flight': self.config.max_particles_in_flight
        }
        
        try:
            mc_engine = MonteCarloEngine(geometry, xs_db, mc_config, decay_db)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                self.logger.error(
                    "GPU out of memory! Try reducing max_particles_in_flight or using CPU"
                )
                raise RuntimeError(
                    f"GPU memory exhausted. Current setting: {self.config.max_particles_in_flight} particles. "
                    f"Try reducing this value or set device='cpu'"
                ) from e
            raise
        
        # Step 6: Initialize dose synthesis
        self.logger.info("Step 6: Initializing dose synthesis")
        dose_synthesis = DoseSynthesis(geometry, self.config.num_batches)
        
        # Step 7: Run Monte Carlo simulation for each nuclide
        self.logger.info("Step 7: Running Monte Carlo simulation")
        
        for nuclide, num_primaries in primaries_per_nuclide.items():
            if num_primaries == 0:
                continue
            
            self.logger.info(f"Simulating {nuclide}...")
            
            try:
                # Simulate nuclide
                dose_map = mc_engine.simulate_nuclide(
                    tia_maps[nuclide],
                    nuclide,
                    num_primaries
                )
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    self.logger.error(
                        f"GPU out of memory during {nuclide} simulation! "
                        f"Try reducing max_particles_in_flight or num_primaries"
                    )
                    raise RuntimeError(
                        f"GPU memory exhausted during {nuclide} simulation. "
                        f"Current: {self.config.max_particles_in_flight} particles, "
                        f"{num_primaries} primaries. Try reducing these values."
                    ) from e
                raise
            
            # Accumulate dose
            dose_synthesis.accumulate_nuclide_dose(nuclide, dose_map)
            
            # For batch uncertainty (simplified: use single batch)
            dose_synthesis.accumulate_batch_dose(dose_map)
            
            # Reset engine for next nuclide
            mc_engine.reset_dose_map()
        
        # Step 8: Export results
        self.logger.info("Step 8: Exporting results")
        results = dose_synthesis.export_results(
            output_format=self.config.output_format,
            output_path=self.config.output_path
        )
        
        # Add statistics
        results['statistics'] = dose_synthesis.get_dose_statistics()
        
        # Performance metrics
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        results['performance'] = {
            'total_time_seconds': elapsed_time,
            'primaries_simulated': self.config.num_primaries,
            'primaries_per_second': self.config.num_primaries / elapsed_time if elapsed_time > 0 else 0
        }
        
        # GPU memory metrics (if CUDA available)
        if self.config.device == 'cuda' and torch.cuda.is_available():
            results['performance']['gpu_memory'] = {
                'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
                'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
                'max_reserved_mb': torch.cuda.max_memory_reserved() / 1024**2
            }
            self.logger.info(
                f"GPU Memory: {results['performance']['gpu_memory']['max_allocated_mb']:.1f} MB peak allocated"
            )
        
        self.logger.info("=" * 60)
        self.logger.info(f"Simulation complete in {elapsed_time:.2f} seconds")
        self.logger.info(f"Primaries/second: {results['performance']['primaries_per_second']:.2e}")
        self.logger.info("=" * 60)
        
        return results
