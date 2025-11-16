#!/usr/bin/env python3
"""
Example: Mask-Based Tissue Definition for Dosimetry Simulation

This example demonstrates how to use segmentation masks to define tissue regions
for accurate dosimetry calculations. This is particularly useful for:
- Organ-specific dosimetry
- Tumor delineation
- Bone segmentation (cortical vs trabecular)
- Integration with treatment planning systems
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import nibabel as nib
import torch

from MCGPURPTDosimetry import DosimetrySimulator, SimulationConfig


def create_synthetic_phantom_with_masks():
    """Create a synthetic phantom with CT and segmentation masks.
    
    This simulates a patient scan with:
    - CT image (HU values)
    - Activity distribution
    - Organ segmentation masks (liver, kidneys, tumor)
    """
    print("Creating synthetic phantom with segmentation masks...")
    
    # Create 64x64x64 phantom
    shape = (64, 64, 64)
    
    # 1. Create CT image (Hounsfield Units)
    ct_data = np.zeros(shape, dtype=np.float32)
    
    # Background: soft tissue (-50 to 50 HU)
    ct_data[:] = np.random.uniform(-50, 50, shape)
    
    # Liver region (center-left): soft tissue (40-60 HU)
    liver_mask = np.zeros(shape, dtype=np.float32)
    liver_mask[20:45, 15:40, 20:45] = 1.0
    ct_data[liver_mask > 0] = np.random.uniform(40, 60, np.sum(liver_mask > 0))
    
    # Kidneys (left and right): soft tissue (30-50 HU)
    kidney_left_mask = np.zeros(shape, dtype=np.float32)
    kidney_left_mask[15:30, 10:20, 25:40] = 1.0
    ct_data[kidney_left_mask > 0] = np.random.uniform(30, 50, np.sum(kidney_left_mask > 0))
    
    kidney_right_mask = np.zeros(shape, dtype=np.float32)
    kidney_right_mask[15:30, 45:55, 25:40] = 1.0
    ct_data[kidney_right_mask > 0] = np.random.uniform(30, 50, np.sum(kidney_right_mask > 0))
    
    # Tumor (in liver): slightly different density (50-70 HU)
    tumor_mask = np.zeros(shape, dtype=np.float32)
    tumor_mask[28:38, 22:32, 28:38] = 1.0
    ct_data[tumor_mask > 0] = np.random.uniform(50, 70, np.sum(tumor_mask > 0))
    
    # Bone region (spine): bone (300-800 HU)
    bone_mask = np.zeros(shape, dtype=np.float32)
    bone_mask[25:40, 28:36, 15:25] = 1.0
    ct_data[bone_mask > 0] = np.random.uniform(300, 800, np.sum(bone_mask > 0))
    
    # 2. Create activity distribution (Bq/voxel)
    activity_data = np.zeros(shape, dtype=np.float32)
    
    # High activity in tumor
    activity_data[tumor_mask > 0] = 1e6  # 1 MBq/voxel
    
    # Moderate activity in liver (background uptake)
    activity_data[liver_mask > 0] += 1e5  # 0.1 MBq/voxel
    
    # Low activity in kidneys (clearance organ)
    activity_data[kidney_left_mask > 0] = 5e5  # 0.5 MBq/voxel
    activity_data[kidney_right_mask > 0] = 5e5
    
    # 3. Create NIfTI images
    affine = np.eye(4)
    affine[0, 0] = 2.0  # 2mm voxel size
    affine[1, 1] = 2.0
    affine[2, 2] = 2.0
    
    ct_img = nib.Nifti1Image(ct_data, affine)
    activity_img = nib.Nifti1Image(activity_data, affine)
    
    # 4. Create segmentation masks
    masks = {
        'Liver': nib.Nifti1Image(liver_mask, affine),
        'Kidney_Left': nib.Nifti1Image(kidney_left_mask, affine),
        'Kidney_Right': nib.Nifti1Image(kidney_right_mask, affine),
        'Tumor': nib.Nifti1Image(tumor_mask, affine),
        'Bone_Cortical': nib.Nifti1Image(bone_mask, affine)
    }
    
    print(f"  ✓ Created phantom: {shape}")
    print(f"  ✓ Voxel size: 2.0 mm")
    print(f"  ✓ Segmented organs: {list(masks.keys())}")
    
    return ct_img, activity_img, masks


def example_1_basic_mask_usage():
    """Example 1: Basic mask-based tissue definition."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Mask-Based Tissue Definition")
    print("=" * 60)
    
    # Create synthetic data
    ct_img, activity_img, masks = create_synthetic_phantom_with_masks()
    
    # Configure simulation
    config = SimulationConfig(
        radionuclide='Lu-177',
        num_primaries=10000,  # Small number for fast demo
        energy_cutoff_keV=10.0,
        num_batches=1,
        output_format='object',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("\nRunning simulation with mask-based tissue definition...")
    print(f"  Radionuclide: {config.radionuclide}")
    print(f"  Primaries: {config.num_primaries}")
    print(f"  Device: {config.device}")
    
    # Run simulation with masks
    simulator = DosimetrySimulator(config)
    results = simulator.run(
        ct_image=ct_img,
        activity_map=activity_img,
        tissue_masks=masks  # Masks override HU-based assignment
    )
    
    # Analyze results
    dose_array = results['total_dose'].get_fdata()
    
    print("\n✓ Simulation complete!")
    print(f"\nDose statistics:")
    print(f"  Mean dose: {np.mean(dose_array[dose_array > 0]):.2e} Gy")
    print(f"  Max dose: {np.max(dose_array):.2e} Gy")
    
    # Organ-specific dose
    for organ_name, mask_img in masks.items():
        mask_data = mask_img.get_fdata()
        organ_dose = dose_array * mask_data
        if np.sum(mask_data > 0) > 0:
            mean_dose = np.mean(organ_dose[mask_data > 0])
            print(f"  {organ_name} mean dose: {mean_dose:.2e} Gy")


def example_2_priority_order():
    """Example 2: Using priority order for overlapping masks."""
    print("\n" + "=" * 60)
    print("Example 2: Priority Order for Overlapping Masks")
    print("=" * 60)
    
    # Create synthetic data
    ct_img, activity_img, masks = create_synthetic_phantom_with_masks()
    
    # Note: Tumor overlaps with Liver
    # Priority order determines which material is assigned in overlap regions
    
    priority_order = ['Liver', 'Kidney_Left', 'Kidney_Right', 'Bone_Cortical', 'Tumor']
    
    print("\nPriority order (last has highest priority):")
    for i, organ in enumerate(priority_order, 1):
        print(f"  {i}. {organ}")
    print("\n  → Tumor will override Liver in overlapping regions")
    
    # Configure simulation
    config = SimulationConfig(
        radionuclide='Lu-177',
        num_primaries=10000,
        output_format='object',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Run simulation with priority order
    simulator = DosimetrySimulator(config)
    results = simulator.run(
        ct_image=ct_img,
        activity_map=activity_img,
        tissue_masks=masks,
        mask_priority_order=priority_order  # Specify priority
    )
    
    print("\n✓ Simulation complete with priority-based mask application!")


def example_3_hybrid_hu_and_masks():
    """Example 3: Hybrid HU-based and mask-based material assignment."""
    print("\n" + "=" * 60)
    print("Example 3: Hybrid HU + Mask-Based Assignment")
    print("=" * 60)
    
    # Create synthetic data
    ct_img, activity_img, masks = create_synthetic_phantom_with_masks()
    
    # Use only specific organ masks, let HU-based assignment handle the rest
    selected_masks = {
        'Tumor': masks['Tumor'],  # Explicitly define tumor
        'Bone_Cortical': masks['Bone_Cortical']  # Explicitly define cortical bone
    }
    
    print("\nUsing hybrid approach:")
    print("  Masked regions:")
    for organ in selected_masks.keys():
        print(f"    - {organ} (explicit material assignment)")
    print("  Unmasked regions:")
    print("    - Assigned based on HU values")
    
    # Configure simulation
    config = SimulationConfig(
        radionuclide='Lu-177',
        num_primaries=10000,
        output_format='object',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Run simulation
    simulator = DosimetrySimulator(config)
    results = simulator.run(
        ct_image=ct_img,
        activity_map=activity_img,
        tissue_masks=selected_masks  # Only tumor and bone explicitly defined
    )
    
    print("\n✓ Simulation complete with hybrid material assignment!")


def example_4_density_handling():
    """Example 4: Density handling options in masked regions."""
    print("\n" + "=" * 60)
    print("Example 4: Density Handling in Masked Regions")
    print("=" * 60)
    
    # Create synthetic data
    ct_img, activity_img, masks = create_synthetic_phantom_with_masks()
    
    print("\nDensity handling options:")
    print("  1. use_ct_density=True (default):")
    print("     - Use CT-derived density in masked regions")
    print("     - Preserves anatomical density variations")
    print("  2. use_ct_density=False:")
    print("     - Use fixed material density from database")
    print("     - Uniform density within each organ")
    
    # Configure simulation
    config = SimulationConfig(
        radionuclide='Lu-177',
        num_primaries=10000,
        output_format='object',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Run with CT-derived density (default)
    print("\nRunning with CT-derived density...")
    simulator = DosimetrySimulator(config)
    results_ct_density = simulator.run(
        ct_image=ct_img,
        activity_map=activity_img,
        tissue_masks=masks,
        use_ct_density=True  # Use CT-derived density
    )
    
    # Run with fixed material density
    print("Running with fixed material density...")
    results_fixed_density = simulator.run(
        ct_image=ct_img,
        activity_map=activity_img,
        tissue_masks=masks,
        use_ct_density=False  # Use database density
    )
    
    print("\n✓ Both simulations complete!")
    print("  → CT-derived density preserves anatomical variations")
    print("  → Fixed density provides uniform organ properties")


def example_5_file_based_masks():
    """Example 5: Loading masks from files."""
    print("\n" + "=" * 60)
    print("Example 5: Loading Masks from Files")
    print("=" * 60)
    
    # Create synthetic data and save to files
    ct_img, activity_img, masks = create_synthetic_phantom_with_masks()
    
    # Create temporary directory
    temp_dir = Path('temp_mask_example')
    temp_dir.mkdir(exist_ok=True)
    
    # Save images to files
    ct_path = temp_dir / 'ct.nii.gz'
    activity_path = temp_dir / 'activity.nii.gz'
    
    nib.save(ct_img, str(ct_path))
    nib.save(activity_img, str(activity_path))
    
    # Save masks to files
    mask_paths = {}
    for organ_name, mask_img in masks.items():
        mask_path = temp_dir / f'{organ_name.lower()}_mask.nii.gz'
        nib.save(mask_img, str(mask_path))
        mask_paths[organ_name] = str(mask_path)
    
    print(f"\nSaved files to: {temp_dir}/")
    print("  CT image: ct.nii.gz")
    print("  Activity map: activity.nii.gz")
    print("  Masks:")
    for organ, path in mask_paths.items():
        print(f"    - {organ}: {Path(path).name}")
    
    # Configure simulation
    config = SimulationConfig(
        radionuclide='Lu-177',
        num_primaries=10000,
        output_format='file',
        output_path=str(temp_dir / 'results'),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Run simulation with file paths
    print("\nRunning simulation with file-based masks...")
    simulator = DosimetrySimulator(config)
    results = simulator.run(
        ct_image=str(ct_path),
        activity_map=str(activity_path),
        tissue_masks=mask_paths  # Dictionary of file paths
    )
    
    print("\n✓ Simulation complete!")
    print(f"  Results saved to: {temp_dir}/results/")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    print(f"  Cleaned up temporary files")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Mask-Based Tissue Definition Examples")
    print("=" * 60)
    print("\nThese examples demonstrate various ways to use segmentation")
    print("masks for accurate tissue definition in dosimetry simulations.")
    
    try:
        # Run examples
        example_1_basic_mask_usage()
        example_2_priority_order()
        example_3_hybrid_hu_and_masks()
        example_4_density_handling()
        example_5_file_based_masks()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
