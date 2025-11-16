"""
Basic usage example for GPU-accelerated dosimetry simulation.

This example demonstrates how to:
1. Load CT and activity images
2. Process geometry
3. Configure and run simulation (placeholder for now)
"""

import torch
import numpy as np
import nibabel as nib
from pathlib import Path

from MCGPURPTDosimetry.core import InputManager, GeometryProcessor
from MCGPURPTDosimetry.utils import SimulationConfig, setup_logger


def create_phantom_data(output_dir: str = './phantom_data'):
    """Create simple phantom data for testing.
    
    Args:
        output_dir: Directory to save phantom data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create a simple 64x64x64 phantom
    shape = (64, 64, 64)
    
    # CT phantom: water sphere in air
    ct_data = np.full(shape, -1000.0, dtype=np.float32)  # Air
    center = np.array(shape) // 2
    
    # Create sphere of water (HU = 0)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                dist = np.sqrt((i-center[0])**2 + (j-center[1])**2 + (k-center[2])**2)
                if dist < 20:  # 20 voxel radius
                    ct_data[i, j, k] = 0.0  # Water
    
    # Activity phantom: uniform activity in sphere
    activity_data = np.zeros(shape, dtype=np.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                dist = np.sqrt((i-center[0])**2 + (j-center[1])**2 + (k-center[2])**2)
                if dist < 20:
                    activity_data[i, j, k] = 1000.0  # 1000 Bq/voxel
    
    # Create affine matrix (1mm isotropic voxels)
    affine = np.eye(4)
    affine[:3, :3] = np.diag([1.0, 1.0, 1.0])
    
    # Save as NIfTI
    ct_img = nib.Nifti1Image(ct_data, affine)
    activity_img = nib.Nifti1Image(activity_data, affine)
    
    ct_path = output_path / 'phantom_ct.nii.gz'
    activity_path = output_path / 'phantom_activity.nii.gz'
    
    nib.save(ct_img, str(ct_path))
    nib.save(activity_img, str(activity_path))
    
    print(f"Phantom data created:")
    print(f"  CT: {ct_path}")
    print(f"  Activity: {activity_path}")
    
    return str(ct_path), str(activity_path)


def example_file_io():
    """Example using file-based I/O."""
    print("\n=== Example 1: File-based I/O ===\n")
    
    try:
        # Set up logging
        logger = setup_logger(level=20)  # INFO level
        
        # Create phantom data
        ct_path, activity_path = create_phantom_data()
    except Exception as e:
        print(f"Error in file I/O example: {e}")
        return None
    
    # Initialize input manager
    input_mgr = InputManager()
    
    # Load images from files
    ct_tensor = input_mgr.load_ct_image(ct_path, device='cpu')
    activity_tensor = input_mgr.load_activity_image(activity_path, device='cpu')
    
    # Validate compatibility
    input_mgr.validate_image_compatibility()
    
    # Get metadata
    voxel_size = input_mgr.get_voxel_dimensions()
    dimensions = input_mgr.get_dimensions()
    affine = input_mgr.get_affine_matrix()
    
    print(f"\nImage loaded:")
    print(f"  Dimensions: {dimensions}")
    print(f"  Voxel size: {voxel_size} mm")
    print(f"  CT range: [{ct_tensor.min():.1f}, {ct_tensor.max():.1f}] HU")
    print(f"  Total activity: {activity_tensor.sum():.2e} Bq")
    
    # Process geometry
    config = SimulationConfig.get_default_config()
    geometry_processor = GeometryProcessor(
        hu_to_material_lut=config.hu_to_material_lut
    )
    
    geometry = geometry_processor.create_geometry_data(
        ct_tensor, voxel_size, affine
    )
    
    print(f"\nGeometry processed:")
    print(f"  Material map shape: {geometry.material_map.shape}")
    print(f"  Density range: [{geometry.density_map.min():.3f}, {geometry.density_map.max():.3f}] g/cm³")
    
    return geometry


def example_object_io():
    """Example using object-based I/O."""
    print("\n=== Example 2: Object-based I/O ===\n")
    
    # Set up logging
    logger = setup_logger(level=20)
    
    # Create phantom data
    ct_path, activity_path = create_phantom_data()
    
    # Load with nibabel first
    ct_img = nib.load(ct_path)
    activity_img = nib.load(activity_path)
    
    print("Images loaded with nibabel")
    
    # Initialize input manager
    input_mgr = InputManager()
    
    # Load from nibabel objects
    ct_tensor = input_mgr.load_ct_image(ct_img, device='cpu')
    activity_tensor = input_mgr.load_activity_image(activity_img, device='cpu')
    
    # Validate compatibility
    input_mgr.validate_image_compatibility()
    
    print(f"\nProcessed from nibabel objects:")
    print(f"  CT tensor shape: {ct_tensor.shape}")
    print(f"  Activity tensor shape: {activity_tensor.shape}")
    
    return ct_tensor, activity_tensor


def example_configuration():
    """Example of configuration management."""
    print("\n=== Example 3: Configuration ===\n")
    
    # Create configuration
    config = SimulationConfig(
        radionuclide='Lu-177',
        num_primaries=100000,
        energy_cutoff_keV=10.0,
        num_batches=10,
        output_format='file',
        output_path='./results/'
    )
    
    print("Configuration created:")
    print(f"  Radionuclide: {config.radionuclide}")
    print(f"  Primaries: {config.num_primaries}")
    print(f"  Energy cutoff: {config.energy_cutoff_keV} keV")
    print(f"  Batches: {config.num_batches}")
    print(f"  Device: {config.device}")
    
    # Save to YAML
    config_path = './phantom_data/config.yaml'
    config.to_yaml(config_path)
    print(f"\nConfiguration saved to: {config_path}")
    
    # Load from YAML
    loaded_config = SimulationConfig.from_yaml(config_path)
    print(f"Configuration loaded from YAML")
    
    return config


if __name__ == '__main__':
    print("GPU-Accelerated Dosimetry System - Basic Usage Examples")
    print("=" * 60)
    
    try:
        # Run examples
        geometry = example_file_io()
        ct_tensor, activity_tensor = example_object_io()
        config = example_configuration()
        
        print("\n" + "=" * 60)
        print("Examples completed successfully!")
        print("\nNote: Full simulation requires physics databases and")
        print("      Monte Carlo engine (tasks 4-13 in implementation plan)")
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user")
    except Exception as e:
        print(f"\n\nError running examples: {e}")
        import traceback
        traceback.print_exc()
