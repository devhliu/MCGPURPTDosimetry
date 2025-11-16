# Tutorials and Examples

## Overview

This document provides step-by-step tutorials for common use cases of MCGPURPTDosimetry, from basic simulations to advanced workflows. Each tutorial includes complete code examples and explanations.

## Tutorial 1: Your First Simulation

### Goal

Run a basic dosimetry simulation with Lu-177 on a simple phantom.

### Prerequisites

- MCGPURPTDosimetry installed
- Test data available (or create synthetic phantom)

### Step 1: Import Required Modules

```python
from MCGPURPTDosimetry import DosimetrySimulator, SimulationConfig
import nibabel as nib
import numpy as np
import torch
```

### Step 2: Create Synthetic Phantom

```python
def create_water_phantom(shape=(64, 64, 64), voxel_size=(2.0, 2.0, 2.0)):
    """Create a simple water phantom (0 HU)."""
    # Water has HU = 0
    ct_data = np.zeros(shape, dtype=np.float32)
    
    # Create affine matrix
    affine = np.eye(4)
    affine[0, 0] = voxel_size[0]
    affine[1, 1] = voxel_size[1]
    affine[2, 2] = voxel_size[2]
    
    # Create NIfTI image
    ct_image = nib.Nifti1Image(ct_data, affine)
    return ct_image

def create_point_source(shape=(64, 64, 64), voxel_size=(2.0, 2.0, 2.0)):
    """Create a point source at center."""
    activity_data = np.zeros(shape, dtype=np.float32)
    
    # Place 1 MBq at center
    center = tuple(s // 2 for s in shape)
    activity_data[center] = 1e6  # 1 MBq = 1e6 Bq
    
    # Create affine matrix
    affine = np.eye(4)
    affine[0, 0] = voxel_size[0]
    affine[1, 1] = voxel_size[1]
    affine[2, 2] = voxel_size[2]
    
    # Create NIfTI image
    activity_image = nib.Nifti1Image(activity_data, affine)
    return activity_image

# Create phantom and source
ct_phantom = create_water_phantom()
activity_map = create_point_source()
```

### Step 3: Configure Simulation

```python
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=100_000,  # Start with small number for testing
    energy_cutoff_keV=10.0,
    num_batches=10,
    output_format='object',  # Return objects, not files
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

print(f"Using device: {config.device}")
```

### Step 4: Run Simulation

```python
# Create simulator
simulator = DosimetrySimulator(config)

# Run simulation
print("Starting simulation...")
results = simulator.run(
    ct_image=ct_phantom,
    activity_map=activity_map
)
print("Simulation complete!")
```

### Step 5: Analyze Results

```python
# Extract dose map
total_dose = results['total_dose'].get_fdata()
uncertainty = results['uncertainty'].get_fdata()

# Print statistics
print(f"\nDose Statistics:")
print(f"  Mean dose: {total_dose.mean():.3f} Gy")
print(f"  Max dose: {total_dose.max():.3f} Gy")
print(f"  Min dose: {total_dose.min():.3f} Gy")

# Print performance
perf = results['performance']
print(f"\nPerformance:")
print(f"  Time: {perf['total_time_seconds']:.2f} seconds")
print(f"  Throughput: {perf['primaries_per_second']:.0f} primaries/second")
```

### Step 6: Visualize Results (Optional)

```python
import matplotlib.pyplot as plt

# Plot central slice
central_slice = total_dose.shape[2] // 2
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(total_dose[:, :, central_slice].T, origin='lower', cmap='hot')
plt.colorbar(label='Dose (Gy)')
plt.title('Dose Distribution')

plt.subplot(1, 2, 2)
plt.imshow(uncertainty[:, :, central_slice].T, origin='lower', cmap='viridis')
plt.colorbar(label='Uncertainty (%)')
plt.title('Uncertainty Map')

plt.tight_layout()
plt.savefig('first_simulation.png')
print("\nPlot saved to first_simulation.png")
```

### Complete Script

```python
# first_simulation.py
from MCGPURPTDosimetry import DosimetrySimulator, SimulationConfig
import nibabel as nib
import numpy as np
import torch

# Create phantom
ct_data = np.zeros((64, 64, 64), dtype=np.float32)
affine = np.diag([2.0, 2.0, 2.0, 1.0])
ct_phantom = nib.Nifti1Image(ct_data, affine)

# Create point source
activity_data = np.zeros((64, 64, 64), dtype=np.float32)
activity_data[32, 32, 32] = 1e6  # 1 MBq
activity_map = nib.Nifti1Image(activity_data, affine)

# Configure and run
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=100_000,
    output_format='object',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

simulator = DosimetrySimulator(config)
results = simulator.run(ct_phantom, activity_map)

# Print results
dose = results['total_dose'].get_fdata()
print(f"Max dose: {dose.max():.3f} Gy")
print(f"Time: {results['performance']['total_time_seconds']:.2f} s")
```

---

## Tutorial 2: Patient-Specific Dosimetry

### Goal

Calculate dose distribution for a real patient case using CT and SPECT images.

### Prerequisites

- Patient CT image (NIfTI format)
- Patient SPECT activity map (NIfTI format, calibrated to Bq/voxel)

### Step 1: Load Patient Images

```python
from MCGPURPTDosimetry import DosimetrySimulator, SimulationConfig
import nibabel as nib

# Load images
ct_image = nib.load('patient_ct.nii.gz')
activity_map = nib.load('patient_spect.nii.gz')

print(f"CT shape: {ct_image.shape}")
print(f"Activity shape: {activity_map.shape}")
print(f"CT voxel size: {ct_image.header.get_zooms()}")
```

### Step 2: Verify Image Compatibility

```python
# Check if images have same dimensions
if ct_image.shape != activity_map.shape:
    print("WARNING: Images have different dimensions!")
    print("Consider resampling activity map to CT grid")
    
# Check voxel sizes
ct_voxel = ct_image.header.get_zooms()
activity_voxel = activity_map.header.get_zooms()

if not np.allclose(ct_voxel, activity_voxel):
    print("WARNING: Images have different voxel sizes!")
```

### Step 3: Configure for Clinical Quality

```python
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=5_000_000,  # 5 million for clinical quality
    energy_cutoff_keV=10.0,
    num_batches=20,  # More batches for better uncertainty
    output_format='file',
    output_path='./patient_results/',
    device='cuda',
    random_seed=42  # For reproducibility
)
```

### Step 4: Run Simulation

```python
simulator = DosimetrySimulator(config)

print("Starting patient dosimetry calculation...")
print("This may take several minutes...")

results = simulator.run(
    ct_image='patient_ct.nii.gz',
    activity_map='patient_spect.nii.gz'
)

print("Calculation complete!")
print(f"Results saved to {config.output_path}")
```

### Step 5: Analyze Organ Doses

```python
# Load results
total_dose = nib.load('./patient_results/total_dose.nii.gz')
dose_data = total_dose.get_fdata()

# Load organ masks (if available)
liver_mask = nib.load('liver_mask.nii.gz').get_fdata().astype(bool)
kidney_mask = nib.load('kidney_mask.nii.gz').get_fdata().astype(bool)
tumor_mask = nib.load('tumor_mask.nii.gz').get_fdata().astype(bool)

# Calculate mean doses
liver_dose = dose_data[liver_mask].mean()
kidney_dose = dose_data[kidney_mask].mean()
tumor_dose = dose_data[tumor_mask].mean()

print(f"\nOrgan Doses:")
print(f"  Liver: {liver_dose:.2f} Gy")
print(f"  Kidney: {kidney_dose:.2f} Gy")
print(f"  Tumor: {tumor_dose:.2f} Gy")
```

### Step 6: Generate Report

```python
import matplotlib.pyplot as plt

# Create multi-panel figure
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Select central slices
z_slice = dose_data.shape[2] // 2

# Plot CT
axes[0, 0].imshow(ct_image.get_fdata()[:, :, z_slice].T, 
                   cmap='gray', origin='lower')
axes[0, 0].set_title('CT Image')

# Plot activity
axes[0, 1].imshow(activity_map.get_fdata()[:, :, z_slice].T, 
                   cmap='hot', origin='lower')
axes[0, 1].set_title('Activity Distribution')

# Plot dose
im = axes[0, 2].imshow(dose_data[:, :, z_slice].T, 
                        cmap='hot', origin='lower')
axes[0, 2].set_title('Dose Distribution')
plt.colorbar(im, ax=axes[0, 2], label='Dose (Gy)')

# Plot uncertainty
uncertainty = nib.load('./patient_results/uncertainty.nii.gz').get_fdata()
im = axes[1, 0].imshow(uncertainty[:, :, z_slice].T, 
                        cmap='viridis', origin='lower')
axes[1, 0].set_title('Uncertainty (%)')
plt.colorbar(im, ax=axes[1, 0])

# Plot dose histogram
axes[1, 1].hist(dose_data[dose_data > 0].flatten(), bins=50)
axes[1, 1].set_xlabel('Dose (Gy)')
axes[1, 1].set_ylabel('Voxel Count')
axes[1, 1].set_title('Dose Histogram')
axes[1, 1].set_yscale('log')

# Plot organ doses
organs = ['Liver', 'Kidney', 'Tumor']
doses = [liver_dose, kidney_dose, tumor_dose]
axes[1, 2].bar(organs, doses)
axes[1, 2].set_ylabel('Mean Dose (Gy)')
axes[1, 2].set_title('Organ Doses')

plt.tight_layout()
plt.savefig('./patient_results/dosimetry_report.png', dpi=300)
print("Report saved to ./patient_results/dosimetry_report.png")
```

---

## Tutorial 3: Using Segmentation Masks

### Goal

Use organ segmentation masks to override CT-based tissue assignment.

### Use Case

When CT imaging has artifacts or when you want to ensure specific tissue properties for organs of interest.

### Step 1: Prepare Segmentation Masks

```python
import nibabel as nib
import numpy as np

# Masks should be binary (0 or 1) or label maps
# Same dimensions as CT image

# Example: Create synthetic masks
shape = (128, 128, 128)

# Tumor mask (sphere at center)
tumor_mask = np.zeros(shape, dtype=np.uint8)
center = np.array(shape) // 2
for i in range(shape[0]):
    for j in range(shape[1]):
        for k in range(shape[2]):
            if np.sqrt((i-center[0])**2 + (j-center[1])**2 + (k-center[2])**2) < 10:
                tumor_mask[i, j, k] = 1

# Save mask
affine = np.eye(4)
nib.save(nib.Nifti1Image(tumor_mask, affine), 'tumor_mask.nii.gz')
```

### Step 2: Configure Tissue Masks

```python
from MCGPURPTDosimetry import DosimetrySimulator, SimulationConfig

# Define tissue masks
tissue_masks = {
    'Tumor': 'tumor_mask.nii.gz',
    'Liver': 'liver_mask.nii.gz',
    'Kidney_Left': 'kidney_left_mask.nii.gz',
    'Kidney_Right': 'kidney_right_mask.nii.gz',
    'Bone_Marrow': 'bone_marrow_mask.nii.gz'
}

# Define priority order (higher priority overrides lower)
mask_priority = ['Tumor', 'Kidney_Left', 'Kidney_Right', 'Liver', 'Bone_Marrow']
```

### Step 3: Run Simulation with Masks

```python
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=1_000_000,
    output_format='file',
    output_path='./masked_results/'
)

simulator = DosimetrySimulator(config)

results = simulator.run(
    ct_image='patient_ct.nii.gz',
    activity_map='patient_activity.nii.gz',
    tissue_masks=tissue_masks,
    mask_priority_order=mask_priority,
    use_ct_density=True  # Use CT-derived density in masked regions
)
```

### Step 4: Verify Mask Application

```python
# Load geometry data to verify material assignment
# (This requires accessing internal data structures)

# Alternative: Visual verification
import matplotlib.pyplot as plt

ct_data = nib.load('patient_ct.nii.gz').get_fdata()
tumor_mask_data = nib.load('tumor_mask.nii.gz').get_fdata()
dose_data = nib.load('./masked_results/total_dose.nii.gz').get_fdata()

z_slice = ct_data.shape[2] // 2

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(ct_data[:, :, z_slice].T, cmap='gray', origin='lower')
axes[0].contour(tumor_mask_data[:, :, z_slice].T, colors='red', linewidths=2)
axes[0].set_title('CT with Tumor Mask')

axes[1].imshow(tumor_mask_data[:, :, z_slice].T, cmap='Reds', origin='lower')
axes[1].set_title('Tumor Mask')

axes[2].imshow(dose_data[:, :, z_slice].T, cmap='hot', origin='lower')
axes[2].contour(tumor_mask_data[:, :, z_slice].T, colors='white', linewidths=1)
axes[2].set_title('Dose with Mask Overlay')

plt.tight_layout()
plt.savefig('./masked_results/mask_verification.png')
```

---

## Tutorial 4: Decay Chain Simulation

### Goal

Simulate dose from Ac-225 including all daughter products in the decay chain.

### Background

Ac-225 undergoes a 6-step decay chain:
```
Ac-225 (α, 10d) → Fr-221 (α, 4.8min) → At-217 (α, 32ms) → 
Bi-213 (α/β, 46min) → Po-213 (α, 4μs) / Tl-209 (β, 2.2min) → Pb-209 (stable)
```

### Step 1: Configure for Decay Chain

```python
from MCGPURPTDosimetry import DosimetrySimulator, SimulationConfig

config = SimulationConfig(
    radionuclide='Ac-225',  # Parent nuclide
    num_primaries=2_000_000,
    output_format='file',
    output_path='./ac225_results/'
)
```

### Step 2: Run Simulation

```python
simulator = DosimetrySimulator(config)

results = simulator.run(
    ct_image='patient_ct.nii.gz',
    activity_map='patient_activity.nii.gz'
)

# Results automatically include all daughters
print("Dose contributions from:")
for nuclide in results['individual_doses'].keys():
    print(f"  - {nuclide}")
```

### Step 3: Analyze Individual Contributions

```python
import nibabel as nib
import numpy as np

# Load individual dose maps
ac225_dose = nib.load('./ac225_results/Ac-225_dose.nii.gz').get_fdata()
fr221_dose = nib.load('./ac225_results/Fr-221_dose.nii.gz').get_fdata()
at217_dose = nib.load('./ac225_results/At-217_dose.nii.gz').get_fdata()
bi213_dose = nib.load('./ac225_results/Bi-213_dose.nii.gz').get_fdata()
total_dose = nib.load('./ac225_results/total_dose.nii.gz').get_fdata()

# Calculate contributions
total_energy = total_dose.sum()
contributions = {
    'Ac-225': ac225_dose.sum() / total_energy * 100,
    'Fr-221': fr221_dose.sum() / total_energy * 100,
    'At-217': at217_dose.sum() / total_energy * 100,
    'Bi-213': bi213_dose.sum() / total_energy * 100,
}

print("\nDose Contributions:")
for nuclide, percent in contributions.items():
    print(f"  {nuclide}: {percent:.1f}%")
```

### Step 4: Visualize Decay Chain Contributions

```python
import matplotlib.pyplot as plt

# Plot contributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

z_slice = total_dose.shape[2] // 2

# Individual nuclides
for idx, (nuclide, dose) in enumerate([
    ('Ac-225', ac225_dose),
    ('Fr-221', fr221_dose),
    ('At-217', at217_dose),
    ('Bi-213', bi213_dose),
    ('Total', total_dose)
]):
    ax = axes[idx // 3, idx % 3]
    im = ax.imshow(dose[:, :, z_slice].T, cmap='hot', origin='lower')
    ax.set_title(f'{nuclide} Dose')
    plt.colorbar(im, ax=ax, label='Dose (Gy)')

# Pie chart of contributions
ax = axes[1, 2]
ax.pie(contributions.values(), labels=contributions.keys(), autopct='%1.1f%%')
ax.set_title('Dose Contributions')

plt.tight_layout()
plt.savefig('./ac225_results/decay_chain_analysis.png')
```

---

## Tutorial 5: Batch Processing Multiple Patients

### Goal

Process multiple patient cases efficiently.

### Step 1: Organize Patient Data

```
patients/
├── patient_001/
│   ├── ct.nii.gz
│   └── activity.nii.gz
├── patient_002/
│   ├── ct.nii.gz
│   └── activity.nii.gz
└── patient_003/
    ├── ct.nii.gz
    └── activity.nii.gz
```

### Step 2: Create Batch Processing Script

```python
from MCGPURPTDosimetry import DosimetrySimulator, SimulationConfig
from pathlib import Path
import json

def process_patient(patient_dir, output_dir, config):
    """Process a single patient."""
    patient_id = patient_dir.name
    print(f"\nProcessing {patient_id}...")
    
    # Create output directory
    patient_output = output_dir / patient_id
    patient_output.mkdir(parents=True, exist_ok=True)
    
    # Update config for this patient
    config.output_path = str(patient_output)
    
    # Run simulation
    simulator = DosimetrySimulator(config)
    
    try:
        results = simulator.run(
            ct_image=str(patient_dir / 'ct.nii.gz'),
            activity_map=str(patient_dir / 'activity.nii.gz')
        )
        
        # Save summary
        summary = {
            'patient_id': patient_id,
            'status': 'success',
            'statistics': results['statistics'],
            'performance': results['performance']
        }
        
        with open(patient_output / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  ✓ {patient_id} complete")
        return summary
        
    except Exception as e:
        print(f"  ✗ {patient_id} failed: {e}")
        return {'patient_id': patient_id, 'status': 'failed', 'error': str(e)}

# Main batch processing
patients_dir = Path('patients')
output_dir = Path('batch_results')
output_dir.mkdir(exist_ok=True)

# Configure simulation
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=1_000_000,
    output_format='file',
    device='cuda'
)

# Process all patients
patient_dirs = sorted(patients_dir.glob('patient_*'))
results = []

for patient_dir in patient_dirs:
    result = process_patient(patient_dir, output_dir, config)
    results.append(result)

# Save batch summary
with open(output_dir / 'batch_summary.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nBatch processing complete!")
print(f"Processed {len(results)} patients")
print(f"Success: {sum(1 for r in results if r['status'] == 'success')}")
print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}")
```

---

## Tutorial 6: Performance Optimization

### Goal

Optimize simulation parameters for speed vs accuracy tradeoff.

### Scenario 1: Rapid Prototyping

```python
# Fast simulation for testing
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=10_000,  # Very few primaries
    energy_cutoff_keV=100.0,  # High cutoff
    num_batches=2,  # Minimal batches
    device='cuda'
)

# Expected: ~1-2 seconds, ±20-30% uncertainty
```

### Scenario 2: Balanced Quality

```python
# Good balance for routine use
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=1_000_000,  # 1 million primaries
    energy_cutoff_keV=10.0,  # Standard cutoff
    num_batches=10,  # Standard batches
    device='cuda'
)

# Expected: ~1-2 minutes, ±5-10% uncertainty
```

### Scenario 3: Publication Quality

```python
# High accuracy for research
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=10_000_000,  # 10 million primaries
    energy_cutoff_keV=1.0,  # Low cutoff
    num_batches=20,  # Many batches
    device='cuda'
)

# Expected: ~10-20 minutes, ±2-5% uncertainty
```

### Memory Optimization

```python
# For limited GPU memory
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=1_000_000,
    max_particles_in_flight=50_000,  # Reduce from default 100k
    device='cuda'
)
```

---

## Common Workflows

### Workflow 1: Treatment Planning

```python
# 1. Load patient images
# 2. Run simulation with planned activity
# 3. Calculate organ doses
# 4. Check against dose constraints
# 5. Adjust activity if needed
# 6. Generate treatment plan report
```

### Workflow 2: Post-Treatment Dosimetry

```python
# 1. Load post-treatment SPECT/PET
# 2. Run simulation with measured activity
# 3. Calculate delivered doses
# 4. Compare with planned doses
# 5. Correlate with treatment response
```

### Workflow 3: Research Study

```python
# 1. Batch process patient cohort
# 2. Extract dose metrics for each patient
# 3. Perform statistical analysis
# 4. Generate publication figures
```

---

**Next**: See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues and solutions.
