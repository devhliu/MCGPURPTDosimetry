"""
Full simulation example demonstrating end-to-end dosimetry calculation.

This example shows:
1. Creating phantom data
2. Generating physics databases
3. Running complete Monte Carlo simulation
4. Analyzing results
"""

import torch
import numpy as np
import nibabel as nib
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from MCGPURPTDosimetry import DosimetrySimulator, SimulationConfig
from MCGPURPTDosimetry.utils import setup_logger


def create_test_phantom(output_dir: str = './test_data'):
    """Create a simple test phantom.
    
    Args:
        output_dir: Directory to save phantom data
        
    Returns:
        Tuple of (ct_path, activity_path)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Creating test phantom...")
    
    # Create a 32x32x32 phantom (small for fast testing)
    shape = (32, 32, 32)
    
    # CT phantom: water sphere in air
    ct_data = np.full(shape, -1000.0, dtype=np.float32)  # Air
    center = np.array(shape) // 2
    
    # Create sphere of water (HU = 0)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                dist = np.sqrt((i-center[0])**2 + (j-center[1])**2 + (k-center[2])**2)
                if dist < 10:  # 10 voxel radius
                    ct_data[i, j, k] = 0.0  # Water
    
    # Activity phantom: uniform activity in sphere
    activity_data = np.zeros(shape, dtype=np.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                dist = np.sqrt((i-center[0])**2 + (j-center[1])**2 + (k-center[2])**2)
                if dist < 10:
                    activity_data[i, j, k] = 1000.0  # 1000 Bq/voxel
    
    # Create affine matrix (2mm isotropic voxels)
    affine = np.eye(4)
    affine[:3, :3] = np.diag([2.0, 2.0, 2.0])
    
    # Save as NIfTI
    ct_img = nib.Nifti1Image(ct_data, affine)
    activity_img = nib.Nifti1Image(activity_data, affine)
    
    ct_path = output_path / 'test_ct.nii.gz'
    activity_path = output_path / 'test_activity.nii.gz'
    
    nib.save(ct_img, str(ct_path))
    nib.save(activity_img, str(activity_path))
    
    print(f"  CT: {ct_path}")
    print(f"  Activity: {activity_path}")
    print(f"  Shape: {shape}, Voxel size: 2x2x2 mm")
    print(f"  Total activity: {np.sum(activity_data):.2e} Bq")
    
    return str(ct_path), str(activity_path)


def run_simulation_example():
    """Run a complete simulation example."""
    print("\n" + "=" * 60)
    print("Full Simulation Example")
    print("=" * 60)
    
    # Set up logging
    logger = setup_logger(level=20)  # INFO level
    
    # Step 1: Create test phantom
    print("\nStep 1: Creating test phantom")
    ct_path, activity_path = create_test_phantom()
    
    # Step 2: Configure simulation
    print("\nStep 2: Configuring simulation")
    config = SimulationConfig(
        radionuclide='Lu-177',
        num_primaries=10000,  # Small number for fast testing
        energy_cutoff_keV=10.0,
        num_batches=5,
        output_format='file',
        output_path='./test_results/',
        device='cpu',  # Use CPU for compatibility
        max_particles_in_flight=10000
    )
    
    print(f"  Radionuclide: {config.radionuclide}")
    print(f"  Primaries: {config.num_primaries}")
    print(f"  Device: {config.device}")
    print(f"  Output: {config.output_path}")
    
    # Step 3: Run simulation
    print("\nStep 3: Running Monte Carlo simulation")
    print("  (This may take a minute...)")
    
    simulator = DosimetrySimulator(config)
    results = simulator.run(
        ct_image=ct_path,
        activity_map=activity_path
    )
    
    # Step 4: Analyze results
    print("\nStep 4: Analyzing results")
    print("\nOutput files:")
    for key, value in results.items():
        if isinstance(value, str) and value.endswith('.nii.gz'):
            print(f"  {key}: {value}")
    
    print("\nDose statistics:")
    if 'statistics' in results:
        for nuclide, stats in results['statistics'].items():
            print(f"\n  {nuclide}:")
            print(f"    Total dose: {stats['total']:.2e} Gy")
            print(f"    Mean dose: {stats['mean']:.2e} Gy")
            print(f"    Max dose: {stats['max']:.2e} Gy")
    
    print("\nPerformance metrics:")
    if 'performance' in results:
        perf = results['performance']
        print(f"  Total time: {perf['total_time_seconds']:.2f} seconds")
        print(f"  Primaries simulated: {perf['primaries_simulated']}")
        print(f"  Primaries/second: {perf['primaries_per_second']:.2e}")
    
    print("\n" + "=" * 60)
    print("Simulation Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {config.output_path}")
    print("\nYou can visualize the dose maps using:")
    print("  - ITK-SNAP")
    print("  - 3D Slicer")
    print("  - nibabel in Python")
    
    return results


def visualize_results_example(results_path: str = './test_results/'):
    """Example of loading and visualizing results.
    
    Args:
        results_path: Path to results directory
    """
    print("\n" + "=" * 60)
    print("Visualization Example")
    print("=" * 60)
    
    results_dir = Path(results_path)
    
    # Load dose map
    dose_file = results_dir / 'total_dose.nii.gz'
    if dose_file.exists():
        print(f"\nLoading dose map: {dose_file}")
        dose_img = nib.load(str(dose_file))
        dose_data = dose_img.get_fdata()
        
        print(f"  Shape: {dose_data.shape}")
        print(f"  Total dose: {np.sum(dose_data):.2e} Gy")
        print(f"  Max dose: {np.max(dose_data):.2e} Gy")
        print(f"  Mean dose (non-zero): {np.mean(dose_data[dose_data > 0]):.2e} Gy")
        
        # Find max dose location
        max_idx = np.unravel_index(np.argmax(dose_data), dose_data.shape)
        print(f"  Max dose location: {max_idx}")
        
        # Simple visualization
        print("\nCentral slice (Z=16):")
        central_slice = dose_data[:, :, 16]
        
        # ASCII visualization (simplified)
        print("  (Dose map - simplified visualization)")
        for i in range(0, 32, 4):
            row = ""
            for j in range(0, 32, 4):
                val = central_slice[i, j]
                if val > 0:
                    row += "█"
                else:
                    row += "·"
            print(f"  {row}")
    
    # Load uncertainty map
    uncertainty_file = results_dir / 'uncertainty.nii.gz'
    if uncertainty_file.exists():
        print(f"\nLoading uncertainty map: {uncertainty_file}")
        unc_img = nib.load(str(uncertainty_file))
        unc_data = unc_img.get_fdata()
        
        nonzero_unc = unc_data[unc_data > 0]
        if len(nonzero_unc) > 0:
            print(f"  Mean uncertainty: {np.mean(nonzero_unc):.2f}%")
            print(f"  Max uncertainty: {np.max(nonzero_unc):.2f}%")


if __name__ == '__main__':
    print("GPU-Accelerated Dosimetry System")
    print("Full Simulation Example")
    print()
    
    # Run simulation
    results = run_simulation_example()
    
    # Visualize results
    if 'total_dose' in results:
        visualize_results_example()
    
    print("\n✓ Example complete!")
