# Troubleshooting Guide

## Overview

This guide helps you diagnose and resolve common issues when using MCGPURPTDosimetry. Issues are organized by category with symptoms, causes, and solutions.

## Installation Issues

### Issue: Package Import Fails

**Symptom**:
```python
>>> from MCGPURPTDosimetry import DosimetrySimulator
ModuleNotFoundError: No module named 'MCGPURPTDosimetry'
```

**Causes**:
1. Package not installed
2. Wrong Python environment
3. Installation failed silently

**Solutions**:

```bash
# Check if package is installed
pip list | grep MCGPURPTDosimetry

# Reinstall package
pip install -e .

# Verify Python environment
which python
python --version

# Check PYTHONPATH
echo $PYTHONPATH
```

### Issue: CUDA Not Available

**Symptom**:
```python
>>> import torch
>>> torch.cuda.is_available()
False
```

**Causes**:
1. No NVIDIA GPU
2. CUDA drivers not installed
3. PyTorch installed without CUDA support
4. Driver/CUDA version mismatch

**Solutions**:

```bash
# Check GPU
nvidia-smi

# Check CUDA version
nvcc --version

# Reinstall PyTorch with CUDA
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: Physics Databases Not Found

**Symptom**:
```
FileNotFoundError: Physics database not found at ...
```

**Causes**:
1. Databases not generated
2. Package data not installed
3. Incorrect database path

**Solutions**:

```bash
# Generate databases
python scripts/generate_minimal_databases.py

# Verify database files exist
ls MCGPURPTDosimetry/physics_data/decay_databases/
ls MCGPURPTDosimetry/physics_data/cross_section_databases/

# Reinstall package with data
pip install -e .
```

## Runtime Errors

### Issue: Out of Memory (GPU)

**Symptom**:
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Causes**:
1. Too many particles in flight
2. Large geometry
3. Other GPU processes
4. Insufficient VRAM

**Solutions**:

**Solution 1: Reduce Particle Count**
```python
config = SimulationConfig(
    max_particles_in_flight=50_000,  # Reduce from default 100k
    num_primaries=500_000  # Reduce if needed
)
```

**Solution 2: Clear GPU Cache**
```python
import torch
torch.cuda.empty_cache()
```

**Solution 3: Close Other GPU Applications**
```bash
# Check GPU usage
nvidia-smi

# Kill other processes if needed
kill <PID>
```

**Solution 4: Use CPU**
```python
config = SimulationConfig(device='cpu')
```

**Solution 5: Downsample Images**
```python
from scipy.ndimage import zoom

# Downsample by factor of 2
ct_data = zoom(ct_data, 0.5)
activity_data = zoom(activity_data, 0.5)
```

### Issue: Out of Memory (CPU)

**Symptom**:
```
MemoryError: Unable to allocate X GiB
```

**Causes**:
1. Large images
2. Insufficient RAM
3. Memory leak

**Solutions**:

```python
# Use smaller data types
ct_data = ct_data.astype(np.float32)  # Instead of float64

# Process in chunks (for batch processing)
for patient in patients:
    process_patient(patient)
    del results  # Free memory
    import gc
    gc.collect()

# Reduce image size
ct_data = ct_data[::2, ::2, ::2]  # Downsample by 2
```

### Issue: Simulation Hangs

**Symptom**:
- Simulation starts but never completes
- No error messages
- GPU/CPU usage drops to zero

**Causes**:
1. Infinite loop in particle transport
2. Deadlock in GPU kernel
3. Very slow progress (appears hung)

**Solutions**:

**Check Progress**:
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Set Timeout** (for testing):
```python
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Simulation timeout")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(300)  # 5 minute timeout

try:
    results = simulator.run(ct, activity)
finally:
    signal.alarm(0)
```

**Reduce Problem Size**:
```python
# Test with minimal case
config = SimulationConfig(
    num_primaries=100,  # Very small
    energy_cutoff_keV=100.0
)
```

### Issue: Incorrect Dose Values

**Symptom**:
- Dose values are NaN, Inf, or negative
- Dose is orders of magnitude wrong
- Dose distribution looks unrealistic

**Causes**:
1. Incorrect activity units
2. Wrong voxel size
3. Image orientation issues
4. Numerical instability

**Solutions**:

**Check Activity Units**:
```python
# Activity should be in Bq/voxel
activity_data = nib.load('activity.nii.gz').get_fdata()
print(f"Activity range: {activity_data.min():.2e} - {activity_data.max():.2e} Bq")

# Convert if needed
activity_data *= 1e6  # MBq to Bq
activity_data *= 1e3  # kBq to Bq
```

**Check Voxel Size**:
```python
# Voxel size should be in mm
voxel_size = ct_image.header.get_zooms()
print(f"Voxel size: {voxel_size} mm")

# If in meters, convert
if voxel_size[0] < 0.1:
    print("WARNING: Voxel size appears to be in meters!")
```

**Check for NaN/Inf**:
```python
import numpy as np

dose_data = results['total_dose'].get_fdata()
print(f"NaN count: {np.isnan(dose_data).sum()}")
print(f"Inf count: {np.isinf(dose_data).sum()}")
print(f"Negative count: {(dose_data < 0).sum()}")

# Replace invalid values
dose_data = np.nan_to_num(dose_data, nan=0.0, posinf=0.0, neginf=0.0)
```

### Issue: Image Compatibility Error

**Symptom**:
```
ValueError: CT and activity images have incompatible dimensions
```

**Causes**:
1. Different image dimensions
2. Different voxel sizes
3. Different orientations

**Solutions**:

**Check Dimensions**:
```python
ct_img = nib.load('ct.nii.gz')
activity_img = nib.load('activity.nii.gz')

print(f"CT shape: {ct_img.shape}")
print(f"Activity shape: {activity_img.shape}")
print(f"CT voxel size: {ct_img.header.get_zooms()}")
print(f"Activity voxel size: {activity_img.header.get_zooms()}")
```

**Resample Activity to CT Grid**:
```python
from scipy.ndimage import zoom

# Calculate zoom factors
zoom_factors = np.array(activity_img.shape) / np.array(ct_img.shape)

# Resample
activity_resampled = zoom(activity_img.get_fdata(), zoom_factors)

# Create new NIfTI with CT affine
activity_new = nib.Nifti1Image(activity_resampled, ct_img.affine)
nib.save(activity_new, 'activity_resampled.nii.gz')
```

## Performance Issues

### Issue: Slow Simulation

**Symptom**:
- Simulation takes much longer than expected
- Low GPU utilization
- Low throughput (primaries/second)

**Causes**:
1. Using CPU instead of GPU
2. Small batch size
3. Frequent CPU-GPU transfers
4. Inefficient geometry

**Solutions**:

**Verify GPU Usage**:
```bash
# Monitor GPU during simulation
watch -n 1 nvidia-smi
```

**Check Device**:
```python
print(f"Using device: {config.device}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

**Increase Batch Size**:
```python
config = SimulationConfig(
    num_primaries=1_000_000,  # Larger batches
    max_particles_in_flight=100_000  # More particles
)
```

**Profile Performance**:
```python
import time

start = time.time()
results = simulator.run(ct, activity)
elapsed = time.time() - start

throughput = config.num_primaries / elapsed
print(f"Throughput: {throughput:.0f} primaries/second")

# Expected: 5000-10000 on GPU, 500-1000 on CPU
```

### Issue: High Uncertainty

**Symptom**:
- Uncertainty >20% in most voxels
- Noisy dose distribution

**Causes**:
1. Too few primaries
2. Too few batches
3. Low activity regions

**Solutions**:

**Increase Primaries**:
```python
config = SimulationConfig(
    num_primaries=10_000_000,  # Increase from 1M
    num_batches=20  # More batches
)
```

**Check Uncertainty Map**:
```python
uncertainty = results['uncertainty'].get_fdata()
print(f"Mean uncertainty: {uncertainty.mean():.1f}%")
print(f"Median uncertainty: {np.median(uncertainty):.1f}%")

# Plot histogram
import matplotlib.pyplot as plt
plt.hist(uncertainty[uncertainty > 0].flatten(), bins=50)
plt.xlabel('Uncertainty (%)')
plt.ylabel('Voxel Count')
plt.show()
```

## Configuration Issues

### Issue: Invalid Configuration

**Symptom**:
```
ValueError: Invalid configuration parameter
```

**Causes**:
1. Typo in parameter name
2. Invalid parameter value
3. Missing required parameter

**Solutions**:

**Check Configuration**:
```python
# Use default configuration as template
config = SimulationConfig.get_default_config()
print(config)

# Modify specific parameters
config.radionuclide = 'Lu-177'
config.num_primaries = 1_000_000
```

**Validate Configuration**:
```python
from MCGPURPTDosimetry.utils.validation import validate_config

try:
    validate_config(config)
    print("Configuration valid")
except ValueError as e:
    print(f"Configuration error: {e}")
```

### Issue: Radionuclide Not Found

**Symptom**:
```
KeyError: 'My-Nuclide' not found in decay database
```

**Causes**:
1. Typo in nuclide name
2. Nuclide not in database
3. Database not loaded

**Solutions**:

**List Available Nuclides**:
```python
from MCGPURPTDosimetry.physics import DecayDatabase

db = DecayDatabase()
print("Available nuclides:")
for nuclide in sorted(db.nuclides.keys()):
    print(f"  - {nuclide}")
```

**Check Spelling**:
```python
# Correct format: Element-Mass
# Examples: Lu-177, Y-90, I-131, Ac-225
```

**Add Custom Nuclide**:
See [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) for adding new radionuclides.

## Data Issues

### Issue: Missing Input Files

**Symptom**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'ct.nii.gz'
```

**Solutions**:

```python
from pathlib import Path

# Check if file exists
ct_path = Path('ct.nii.gz')
if not ct_path.exists():
    print(f"File not found: {ct_path.absolute()}")
    
# Use absolute paths
ct_path = Path('/full/path/to/ct.nii.gz')
```

### Issue: Corrupted NIfTI Files

**Symptom**:
```
nibabel.filebasedimages.ImageFileError: Cannot read file header
```

**Solutions**:

```python
import nibabel as nib

# Try to load and validate
try:
    img = nib.load('ct.nii.gz')
    print(f"Shape: {img.shape}")
    print(f"Data type: {img.get_data_dtype()}")
    
    # Check for corruption
    data = img.get_fdata()
    print(f"Data range: {data.min()} - {data.max()}")
    
except Exception as e:
    print(f"File corrupted: {e}")
    print("Try re-exporting from source")
```

## Debugging Strategies

### Enable Verbose Logging

```python
import logging

# Set logging level
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Run simulation
results = simulator.run(ct, activity)
```

### Use Minimal Test Case

```python
# Create minimal phantom
ct_data = np.zeros((32, 32, 32), dtype=np.float32)
activity_data = np.zeros((32, 32, 32), dtype=np.float32)
activity_data[16, 16, 16] = 1e6

# Minimal configuration
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=100,
    output_format='object',
    device='cpu'  # Use CPU for debugging
)

# Run and check
results = simulator.run(ct_phantom, activity_phantom)
print("Minimal test passed!")
```

### Check Intermediate Results

```python
# Access intermediate data (requires modifying code)
from MCGPURPTDosimetry.core import InputManager, GeometryProcessor

# Test input loading
input_mgr = InputManager()
ct_tensor = input_mgr.load_ct_image('ct.nii.gz', device='cpu')
print(f"CT loaded: {ct_tensor.shape}")

# Test geometry processing
geom_proc = GeometryProcessor()
geometry = geom_proc.create_geometry_data(ct_tensor, ...)
print(f"Materials: {geometry.material_map.unique()}")
```

### Profile Memory Usage

```python
import torch

# Reset memory stats
torch.cuda.reset_peak_memory_stats()

# Run simulation
results = simulator.run(ct, activity)

# Check memory usage
print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
print(f"Current memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
```

## Getting Help

### Before Asking for Help

1. **Check this guide** for similar issues
2. **Search GitHub issues** for existing reports
3. **Try minimal test case** to isolate problem
4. **Collect diagnostic information** (see below)

### Diagnostic Information to Provide

```python
import sys
import torch
import nibabel as nib
import MCGPURPTDosimetry

print("=== System Information ===")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"nibabel version: {nib.__version__}")
print(f"MCGPURPTDosimetry version: {MCGPURPTDosimetry.__version__}")

print("\n=== GPU Information ===")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

print("\n=== Configuration ===")
print(config)

print("\n=== Error Message ===")
# Include full error traceback
```

### Reporting Bugs

Create GitHub issue with:

1. **Title**: Brief description of problem
2. **Description**: What you expected vs what happened
3. **Reproduction**: Minimal code to reproduce
4. **Environment**: Diagnostic information above
5. **Logs**: Relevant log output
6. **Data**: Sample data if applicable (anonymized)

### Example Bug Report

```markdown
## Bug: Out of memory with small phantom

### Description
Simulation fails with CUDA out of memory error even with small 64³ phantom.

### Expected Behavior
Should run successfully with <1 GB memory.

### Actual Behavior
RuntimeError: CUDA out of memory. Tried to allocate 4.00 GiB

### Reproduction
```python
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=100_000,
    device='cuda'
)
simulator = DosimetrySimulator(config)
results = simulator.run(ct_64, activity_64)
```

### Environment
- OS: Ubuntu 20.04
- Python: 3.10.12
- PyTorch: 2.0.1+cu118
- GPU: NVIDIA RTX 3060 (12 GB)
- MCGPURPTDosimetry: 0.1.0

### Logs
[Attach simulation.log]

### Workaround
Using device='cpu' works but is slow.
```

---

**Next**: See [FAQ.md](FAQ.md) for frequently asked questions.
