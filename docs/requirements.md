# System Requirements and Dependencies

## Overview

This document specifies the hardware, software, and dependency requirements for MCGPURPTDosimetry, along with installation instructions and troubleshooting guidance.

## Hardware Requirements

### Minimum Requirements

**CPU**:
- Modern x86-64 processor (Intel Core i5 or AMD Ryzen 5 equivalent)
- 4 cores recommended for parallel data processing

**Memory**:
- 8 GB RAM minimum
- 16 GB RAM recommended for large patient datasets (512³ voxels)

**Storage**:
- 2 GB for software and physics databases
- 10-50 GB for patient data and simulation results

**GPU** (Optional but Recommended):
- NVIDIA GPU with CUDA Compute Capability ≥ 3.5
- 4 GB VRAM minimum
- 8 GB VRAM recommended for large geometries

### Recommended Configuration

**For Clinical Use**:
- Intel Core i7/i9 or AMD Ryzen 7/9
- 32 GB RAM
- NVIDIA RTX 3060 or better (12 GB VRAM)
- SSD storage for fast I/O

**For Research/Development**:
- Workstation-class CPU (Intel Xeon, AMD Threadripper)
- 64 GB RAM
- NVIDIA RTX 4090 or A6000 (24 GB VRAM)
- NVMe SSD storage

### GPU Compatibility

**Supported NVIDIA GPUs**:
- **Consumer**: GTX 1060 and newer, RTX 20/30/40 series
- **Professional**: Quadro P4000 and newer, RTX A-series
- **Data Center**: Tesla V100, A100, H100

**Not Supported**:
- AMD GPUs (no CUDA support)
- Intel GPUs (no CUDA support)
- Apple Silicon (no CUDA support, but CPU mode works)

**CPU Fallback**:
- All functionality available on CPU (10-20x slower)
- Useful for testing, small phantoms, or systems without GPU

## Software Requirements

### Operating System

**Supported**:
- **Linux**: Ubuntu 20.04+, CentOS 8+, Debian 11+
- **Windows**: Windows 10/11 (64-bit)
- **macOS**: macOS 11+ (CPU only, no CUDA support)

**Recommended**: Linux for best performance and stability

### Python Environment

**Python Version**:
- Python 3.8, 3.9, 3.10, or 3.11
- Python 3.10 recommended for best compatibility

**Package Manager**:
- pip (included with Python)
- conda/mamba (optional, for environment management)

### CUDA Toolkit

**For GPU Acceleration**:
- CUDA Toolkit 11.7 or newer
- CUDA 12.x recommended for latest GPUs
- cuDNN not required (PyTorch handles this)

**Installation**:
- Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- Or install via conda: `conda install cuda -c nvidia`

**Verification**:
```bash
nvcc --version  # Check CUDA compiler
nvidia-smi      # Check GPU driver and CUDA version
```

## Python Dependencies

### Core Dependencies

Listed in `requirements.txt`:

```
torch>=2.0.0
nibabel>=3.0.0
numpy>=1.20.0
h5py>=3.0.0
pyyaml>=5.0.0
```

### Dependency Details

#### PyTorch (≥2.0.0)

**Purpose**: GPU tensor operations, CUDA integration, automatic differentiation

**Installation**:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Version Notes**:
- PyTorch 2.0+ required for improved performance and stability
- PyTorch 2.1+ recommended for latest optimizations

#### nibabel (≥3.0.0)

**Purpose**: Medical image I/O (NIfTI format)

**Installation**:
```bash
pip install nibabel
```

**Features Used**:
- NIfTI file reading/writing
- Affine matrix handling
- Image resampling utilities

#### NumPy (≥1.20.0)

**Purpose**: Array operations, CPU-side computations

**Installation**:
```bash
pip install numpy
```

**Version Notes**:
- NumPy 1.20+ required for modern array API
- NumPy 1.24+ recommended

#### h5py (≥3.0.0)

**Purpose**: HDF5 file I/O for cross-section databases

**Installation**:
```bash
pip install h5py
```

**Features Used**:
- Reading physics databases
- Efficient binary data storage

#### PyYAML (≥5.0.0)

**Purpose**: Configuration file parsing

**Installation**:
```bash
pip install pyyaml
```

**Features Used**:
- YAML configuration loading
- Configuration serialization

### Optional Dependencies

#### Development Tools

```bash
pip install pytest pytest-cov black flake8 mypy
```

- **pytest**: Unit testing framework
- **pytest-cov**: Code coverage analysis
- **black**: Code formatting
- **flake8**: Linting
- **mypy**: Type checking

#### Visualization

```bash
pip install matplotlib scipy
```

- **matplotlib**: Plotting dose distributions
- **scipy**: Image processing, interpolation

#### Documentation

```bash
pip install sphinx sphinx-rtd-theme
```

- **sphinx**: Documentation generation
- **sphinx-rtd-theme**: ReadTheDocs theme

## Installation

### Quick Install (Recommended)

```bash
# Clone repository
git clone https://github.com/yourusername/MCGPURPTDosimetry.git
cd MCGPURPTDosimetry

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .

# Verify installation
python -c "from MCGPURPTDosimetry import DosimetrySimulator; print('✓ Installation successful')"
```

### Conda Environment (Alternative)

```bash
# Create conda environment
conda create -n dosimetry python=3.10
conda activate dosimetry

# Install PyTorch with CUDA
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install nibabel h5py pyyaml

# Install package
pip install -e .
```

### Docker Installation (Isolated Environment)

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip
COPY requirements.txt /app/
RUN pip3 install -r /app/requirements.txt
COPY . /app/
WORKDIR /app
RUN pip3 install -e .

CMD ["python3"]
```

```bash
# Build and run
docker build -t mcgpu-dosimetry .
docker run --gpus all -it mcgpu-dosimetry
```

## Verification

### Check GPU Availability

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("Running in CPU mode")
```

### Run Test Simulation

```python
from MCGPURPTDosimetry import DosimetrySimulator, SimulationConfig
import torch

# Create minimal test configuration
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=1000,
    output_format='object',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

print("Configuration validated successfully")
print(f"Using device: {config.device}")
```

### Check Physics Databases

```python
from MCGPURPTDosimetry.physics import DecayDatabase, CrossSectionDatabase

# Check decay database
decay_db = DecayDatabase()
print(f"Decay database loaded: {len(decay_db.nuclides)} nuclides")

# Check cross-section database
xs_db = CrossSectionDatabase()
print(f"Cross-section database loaded: {len(xs_db.materials)} materials")
```

## Troubleshooting

### CUDA Not Available

**Symptom**: `torch.cuda.is_available()` returns `False`

**Solutions**:
1. Check GPU driver: `nvidia-smi`
2. Reinstall PyTorch with correct CUDA version
3. Verify CUDA toolkit installation: `nvcc --version`
4. Check GPU compatibility (Compute Capability ≥ 3.5)

### Out of Memory Errors

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce `max_particles_in_flight` in configuration
2. Reduce `num_primaries` or process in batches
3. Use smaller geometry (downsample CT/activity images)
4. Close other GPU applications
5. Fall back to CPU mode

### Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'MCGPURPTDosimetry'`

**Solutions**:
1. Ensure package is installed: `pip install -e .`
2. Check Python environment: `which python`
3. Verify PYTHONPATH includes package directory

### Physics Database Not Found

**Symptom**: `FileNotFoundError: Physics database not found`

**Solutions**:
1. Generate databases: `python scripts/generate_minimal_databases.py`
2. Check database paths in configuration
3. Verify package data installation: `pip install -e .`

### Slow Performance

**Symptom**: Simulation runs slower than expected

**Solutions**:
1. Verify GPU is being used: Check `nvidia-smi` during simulation
2. Increase `num_primaries` for better GPU utilization
3. Check for CPU bottlenecks (data loading, preprocessing)
4. Profile code to identify hotspots

## Performance Benchmarks

### Expected Performance

**GPU (NVIDIA RTX 3080)**:
- Small phantom (64³): ~2 seconds for 10,000 primaries
- Medium phantom (128³): ~10 seconds for 100,000 primaries
- Large phantom (256³): ~60 seconds for 1,000,000 primaries

**CPU (Intel i7-10700K)**:
- Small phantom (64³): ~20 seconds for 10,000 primaries
- Medium phantom (128³): ~120 seconds for 100,000 primaries
- Large phantom (256³): ~600 seconds for 1,000,000 primaries

**Speedup**: 10-20x GPU vs CPU

### Memory Usage

**GPU Memory**:
- Base overhead: ~500 MB (PyTorch, CUDA)
- Geometry (256³): ~256 MB
- Particle stacks (100k particles): ~300 MB
- Physics databases: ~50 MB
- **Total**: ~1.1 GB for typical simulation

**System Memory**:
- Python interpreter: ~100 MB
- Medical images (256³): ~128 MB
- Results storage: ~256 MB
- **Total**: ~500 MB for typical simulation

## Platform-Specific Notes

### Linux

- Best performance and stability
- Native CUDA support
- Recommended for production use

### Windows

- Fully supported with CUDA
- May require Visual Studio C++ redistributables
- WSL2 supported for Linux-like environment

### macOS

- CPU mode only (no CUDA)
- Apple Silicon (M1/M2) supported via CPU
- Performance limited compared to GPU

## Upgrade Path

### From Version 0.1.x to 0.2.x

```bash
# Update package
git pull origin main
pip install -e . --upgrade

# Regenerate physics databases if schema changed
python scripts/generate_minimal_databases.py
```

### Dependency Updates

```bash
# Update all dependencies
pip install -r requirements.txt --upgrade

# Update PyTorch specifically
pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu118
```

## Support and Resources

### Documentation

- [User Guide](USER_GUIDE.md)
- [API Reference](API_REFERENCE.md)
- [Physics Documentation](../PHYSICS.md)

### Community

- GitHub Issues: Report bugs and request features
- Discussions: Ask questions and share experiences

### Commercial Support

Contact the development team for:
- Custom feature development
- Performance optimization
- Clinical validation support
- Training and consultation

---

**Next**: See [ARCHITECTURE.md](ARCHITECTURE.md) for system design and implementation details.
