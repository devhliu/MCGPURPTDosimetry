# User Guide

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Physics Data Preparation](#physics-data-preparation)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0 with CUDA support (for GPU acceleration)
- nibabel ≥ 3.0
- numpy ≥ 1.20
- h5py ≥ 3.0
- pyyaml ≥ 5.0

### Installation Steps

#### 1. Install PyTorch with CUDA

Visit [pytorch.org](https://pytorch.org) and install PyTorch with CUDA support for your system.

Example for CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. Install MCGPURPTDosimetry

```bash
# Clone repository
git clone https://github.com/devhliu/MCGPURPTDosimetry.git
cd MCGPURPTDosimetry

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

#### 3. Verify Installation

```python
import torch
from MCGPURPTDosimetry import DosimetrySimulator, SimulationConfig

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Test basic functionality
config = SimulationConfig.get_default_config()
print("✓ Installation successful!")
```

### Troubleshooting Installation

**Issue**: CUDA not available
- **Solution**: Install PyTorch with CUDA support (see step 1)
- **Fallback**: Use CPU mode (slower but functional)

**Issue**: Import errors
- **Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

**Issue**: Version conflicts
- **Solution**: Create a fresh virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  pip install -r requirements.txt
  ```

---

## Quick Start

### Minimal Example

```python
from MCGPURPTDosimetry import DosimetrySimulator, SimulationConfig

# Configure simulation
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=1_000_000,
    output_format='file',
    output_path='./results/'
)

# Run simulation
simulator = DosimetrySimulator(config)
results = simulator.run(
    ct_image='patient_ct.nii.gz',
    activity_map='patient_activity.nii.gz'
)

print(f"Simulation complete!")
print(f"Results saved to: {config.output_path}")
```

### What Happens

1. **Configuration**: Set up simulation parameters (radionuclide, number of particles, output)
2. **Initialization**: Load physics databases and prepare geometry
3. **Simulation**: Run Monte Carlo particle transport on GPU
4. **Output**: Save dose maps and statistics

---

## Basic Usage

### 1. File-Based I/O

Use file paths for input and output:

```python
from MCGPURPTDosimetry import DosimetrySimulator, SimulationConfig

# Configure for file I/O
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=1_000_000,
    output_format='file',
    output_path='./results/'
)

# Run simulation
simulator = DosimetrySimulator(config)
results = simulator.run(
    ct_image='patient_ct.nii.gz',
    activity_map='patient_activity.nii.gz'
)

# Results saved to files:
# - ./results/total_dose.nii.gz
# - ./results/uncertainty.nii.gz
# - ./results/Lu-177_dose.nii.gz
```

### 2. Object-Based I/O

Use nibabel objects for integration with Python workflows:

```python
import nibabel as nib
from MCGPURPTDosimetry import DosimetrySimulator, SimulationConfig

# Load images in Python
ct_img = nib.load('patient_ct.nii.gz')
activity_img = nib.load('patient_activity.nii.gz')

# Configure for object output
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=1_000_000,
    output_format='object'
)

# Run simulation
simulator = DosimetrySimulator(config)
results = simulator.run(
    ct_image=ct_img,
    activity_map=activity_img
)

# Results are nibabel objects
total_dose_img = results['total_dose']
uncertainty_img = results['uncertainty']

# Continue processing in Python
dose_array = total_dose_img.get_fdata()
print(f"Mean dose: {dose_array.mean():.2e} Gy")
```

### 3. Configuration Management

#### Save Configuration

```python
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=1_000_000,
    energy_cutoff_keV=10.0
)

# Save to YAML
config.to_yaml('simulation_config.yaml')
```

#### Load Configuration

```python
# Load from YAML
config = SimulationConfig.from_yaml('simulation_config.yaml')

# Run simulation with loaded config
simulator = DosimetrySimulator(config)
results = simulator.run(ct_image='ct.nii.gz', activity_map='activity.nii.gz')
```

### 4. Analyzing Results

```python
import numpy as np

# Get dose array
dose_array = results['total_dose'].get_fdata()

# Basic statistics
print(f"Mean dose: {np.mean(dose_array[dose_array > 0]):.2e} Gy")
print(f"Max dose: {np.max(dose_array):.2e} Gy")
print(f"Min dose: {np.min(dose_array[dose_array > 0]):.2e} Gy")

# Dose-volume histogram
dose_values = dose_array[dose_array > 0].flatten()
hist, bins = np.histogram(dose_values, bins=100)

# Performance metrics
print(f"\nPerformance:")
print(f"  Total time: {results['performance']['total_time_seconds']:.2f} s")
print(f"  Primaries/second: {results['performance']['primaries_per_second']:.2e}")
```

---

## Advanced Features

### 1. Mask-Based Tissue Definition

Use segmentation masks to explicitly define tissue regions:

```python
import nibabel as nib

# Load segmentation masks
liver_mask = nib.load('segmentation/liver.nii.gz')
tumor_mask = nib.load('segmentation/tumor.nii.gz')
kidney_left_mask = nib.load('segmentation/kidney_left.nii.gz')
kidney_right_mask = nib.load('segmentation/kidney_right.nii.gz')

# Create mask dictionary
tissue_masks = {
    'Liver': liver_mask,
    'Tumor': tumor_mask,
    'Kidney_Left': kidney_left_mask,
    'Kidney_Right': kidney_right_mask
}

# Run simulation with masks
results = simulator.run(
    ct_image=ct_img,
    activity_map=activity_img,
    tissue_masks=tissue_masks
)
```

**See also**: [Mask-Based Workflow Guide](MASK_BASED_WORKFLOW.md)

### 2. Multi-Nuclide Simulations

Simulate decay chains with multiple radionuclides:

```python
# Ac-225 decay chain simulation
# Ac-225 → Fr-221 → At-217 → Bi-213 → Po-213/Tl-209

config = SimulationConfig(
    radionuclide='Ac-225',  # Parent nuclide
    num_primaries=1_000_000
)

simulator = DosimetrySimulator(config)
results = simulator.run(ct_image=ct_img, activity_map=activity_img)

# Results include contributions from all daughters
# - Ac-225_dose.nii.gz
# - Fr-221_dose.nii.gz
# - At-217_dose.nii.gz
# - Bi-213_dose.nii.gz
# - Po-213_dose.nii.gz
# - total_dose.nii.gz (sum of all)
```

### 3. Custom Physics Databases

Use custom decay or cross-section databases:

```python
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=1_000_000,
    decay_database_path='custom/decay_database.json',
    cross_section_database_path='custom/cross_sections.h5'
)
```

### 4. Reproducible Simulations

Set random seed for reproducibility:

```python
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=1_000_000,
    random_seed=42  # Fixed seed for reproducibility
)

# Multiple runs will produce identical results
```

### 5. Uncertainty Quantification

Control uncertainty estimation with batch processing:

```python
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=1_000_000,
    num_batches=10  # More batches = better uncertainty estimate
)

results = simulator.run(ct_image=ct_img, activity_map=activity_img)

# Uncertainty map available
uncertainty_array = results['uncertainty'].get_fdata()
relative_uncertainty = uncertainty_array / results['total_dose'].get_fdata()
```

---

## Physics Data Preparation

### Generating Decay Databases

Create custom decay databases from ICRP-107 data:

```python
from MCGPURPTDosimetry.physics_data_preparation import DecayDatabaseGenerator

# Initialize generator
generator = DecayDatabaseGenerator('path/to/icrp107_data/')

# Parse nuclides
nuclides = ['Lu-177', 'Y-90', 'I-131', 'F-18', 'Ga-68']
generator.parse_icrp107(nuclides)

# Generate database
generator.generate_database('custom/decay_database.json')

# Validate
is_valid = generator.validate_database('custom/decay_database.json')
print(f"Database valid: {is_valid}")
```

### Generating Cross-Section Databases

Create custom cross-section databases:

```python
import numpy as np
from MCGPURPTDosimetry.physics_data_preparation import CrossSectionGenerator

# Initialize generator
xs_gen = CrossSectionGenerator(physics_backend='geant4')

# Define materials
xs_gen.define_material(
    'Soft_Tissue',
    composition={'H': 0.105, 'C': 0.256, 'N': 0.027, 'O': 0.602},
    density=1.04
)

xs_gen.define_material(
    'Bone',
    composition={'H': 0.064, 'C': 0.278, 'N': 0.027, 'O': 0.410, 
                 'P': 0.070, 'Ca': 0.147},
    density=1.85
)

# Calculate cross-sections
energy_grid = np.logspace(1, 7, 1000)  # 10 eV to 10 MeV
xs_gen.calculate_cross_sections(energy_grid, ['Soft_Tissue', 'Bone'])

# Export database
xs_gen.export_database('custom/cross_sections.h5')
```

### Using Custom Databases

```python
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=1_000_000,
    decay_database_path='custom/decay_database.json',
    cross_section_database_path='custom/cross_sections.h5'
)

simulator = DosimetrySimulator(config)
results = simulator.run(ct_image=ct_img, activity_map=activity_img)
```

---

## Performance Optimization

### GPU Acceleration

#### Optimal Configuration

```python
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=10_000_000,  # Large number for GPU
    max_particles_in_flight=100_000,  # Adjust based on GPU memory
    device='cuda'
)
```

#### Memory Management

```python
# For limited GPU memory, reduce particles in flight
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=1_000_000,
    max_particles_in_flight=50_000,  # Reduce if out of memory
    device='cuda'
)
```

#### Check GPU Usage

```python
import torch

# Before simulation
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Run simulation
results = simulator.run(ct_image=ct_img, activity_map=activity_img)

# After simulation
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

### CPU Fallback

For systems without CUDA:

```python
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=100_000,  # Reduce for CPU
    device='cpu'
)

# Simulation will run on CPU (slower but functional)
```

### Batch Processing

Process multiple patients efficiently:

```python
import glob

# Get all patient files
ct_files = sorted(glob.glob('patients/*/ct.nii.gz'))
activity_files = sorted(glob.glob('patients/*/activity.nii.gz'))

# Process in batch
for ct_file, activity_file in zip(ct_files, activity_files):
    patient_id = ct_file.split('/')[1]
    
    config = SimulationConfig(
        radionuclide='Lu-177',
        num_primaries=1_000_000,
        output_format='file',
        output_path=f'results/{patient_id}/'
    )
    
    simulator = DosimetrySimulator(config)
    results = simulator.run(ct_image=ct_file, activity_map=activity_file)
    
    print(f"✓ Processed patient {patient_id}")
```

### Performance Benchmarking

```python
import time

# Benchmark different configurations
configs = [
    {'num_primaries': 100_000, 'device': 'cuda'},
    {'num_primaries': 1_000_000, 'device': 'cuda'},
    {'num_primaries': 10_000_000, 'device': 'cuda'},
]

for cfg in configs:
    config = SimulationConfig(radionuclide='Lu-177', **cfg)
    simulator = DosimetrySimulator(config)
    
    start = time.time()
    results = simulator.run(ct_image=ct_img, activity_map=activity_img)
    elapsed = time.time() - start
    
    primaries_per_sec = cfg['num_primaries'] / elapsed
    print(f"Primaries: {cfg['num_primaries']:,} | "
          f"Time: {elapsed:.2f}s | "
          f"Rate: {primaries_per_sec:.2e} primaries/s")
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce `max_particles_in_flight`:
  ```python
  config = SimulationConfig(
      radionuclide='Lu-177',
      num_primaries=1_000_000,
      max_particles_in_flight=50_000  # Reduce from default 100_000
  )
  ```
- Use smaller image sizes
- Switch to CPU mode (slower):
  ```python
  config = SimulationConfig(radionuclide='Lu-177', device='cpu')
  ```

#### 2. Image Dimension Mismatch

**Error**: `ImageDimensionMismatchError: CT and activity images have different dimensions`

**Solution**: Resample images to same dimensions:
```python
from scipy.ndimage import zoom

# Resample activity to match CT
ct_shape = ct_img.get_fdata().shape
activity_data = activity_img.get_fdata()

zoom_factors = [ct_shape[i] / activity_data.shape[i] for i in range(3)]
activity_resampled = zoom(activity_data, zoom_factors, order=1)

# Create new image
activity_img_resampled = nib.Nifti1Image(activity_resampled, ct_img.affine)
```

#### 3. Unknown Material Error

**Error**: `InvalidMaterialError: Unknown tissue type: 'CustomTissue'`

**Solution**: Use valid material names:
```python
# Valid materials:
valid_materials = [
    'Air', 'Lung', 'Muscle', 'Soft_Tissue', 'Fat',
    'Bone_Cortical', 'Bone_Trabecular', 'Bone_Generic',
    'Bone', 'Water', 'Iodine_Contrast_Mixture'
]

# Use Soft_Tissue for custom organs
tissue_masks = {
    'Liver': liver_mask,  # Uses Soft_Tissue properties
    'Tumor': tumor_mask   # Uses Soft_Tissue properties
}
```

#### 4. Slow Performance

**Issue**: Simulation is very slow

**Solutions**:
- Ensure CUDA is available:
  ```python
  import torch
  print(f"CUDA available: {torch.cuda.is_available()}")
  ```
- Reduce number of primaries for testing:
  ```python
  config = SimulationConfig(num_primaries=10_000)  # Fast test
  ```
- Check GPU utilization:
  ```bash
  nvidia-smi
  ```

#### 5. Unexpected Dose Distribution

**Issue**: Dose distribution looks incorrect

**Checks**:
1. Verify activity units (Bq/voxel)
2. Check CT HU values are correct
3. Validate mask alignment with CT
4. Review material assignment:
   ```python
   # Check material map
   geometry = geom_processor.create_geometry_data(ct_tensor, ...)
   material_map = geometry.material_map.cpu().numpy()
   
   # Visualize
   import matplotlib.pyplot as plt
   plt.imshow(material_map[:, :, slice_idx])
   plt.colorbar()
   plt.show()
   ```

### Getting Help

1. **Check documentation**:
   - [API Reference](API_REFERENCE.md)
   - [Mask-Based Workflow](MASK_BASED_WORKFLOW.md)
   - [Examples](../examples/)

2. **Review examples**:
   - `examples/basic_usage.py`
   - `examples/full_simulation_example.py`
   - `examples/mask_based_tissue_definition.py`

3. **Enable debug logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

4. **Open an issue**:
   - Provide error message
   - Include minimal reproducible example
   - Specify system configuration (OS, Python version, PyTorch version, CUDA version)

---

## Next Steps

- **Learn advanced features**: [Mask-Based Workflow](MASK_BASED_WORKFLOW.md)
- **Explore API**: [API Reference](API_REFERENCE.md)
- **Run examples**: See `examples/` directory
- **Customize physics**: Generate custom databases
- **Optimize performance**: Tune configuration for your hardware

---

## Additional Resources

- **Design Document**: `.kiro/specs/gpu-dosimetry-system/design.md`
- **Requirements**: `.kiro/specs/gpu-dosimetry-system/requirements.md`
- **GitHub Repository**: [Link to repository]
- **Issue Tracker**: [Link to issues]
- **Discussions**: [Link to discussions]
