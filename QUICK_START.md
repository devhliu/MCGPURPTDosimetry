# Quick Start Guide

## Installation

### 1. Clone and Install

```bash
git clone <repository-url>
cd MCGPURPTDosimetry
pip install -r requirements.txt
pip install -e .
```

### 2. Verify Installation

```bash
python -c "from MCGPURPTDosimetry import DosimetrySimulator; print('✓ Installation successful')"
```

## Basic Usage

### Example 1: File-Based I/O

```python
from MCGPURPTDosimetry import DosimetrySimulator, SimulationConfig

# Configure simulation
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=1_000_000,
    energy_cutoff_keV=10.0,
    num_batches=10,
    output_format='file',
    output_path='./results/'
)

# Run simulation
simulator = DosimetrySimulator(config)
results = simulator.run(
    ct_image='patient_ct.nii.gz',
    activity_map='patient_activity.nii.gz'
)

# Results saved to ./results/
# - total_dose.nii.gz
# - Lu-177_dose.nii.gz
# - uncertainty.nii.gz
# - simulation.log
```

### Example 2: Object-Based I/O (In-Memory)

```python
import nibabel as nib
from MCGPURPTDosimetry import DosimetrySimulator, SimulationConfig

# Load images as nibabel objects
ct_img = nib.load('patient_ct.nii.gz')
activity_img = nib.load('patient_activity.nii.gz')

# Configure for object output
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=500_000,
    output_format='object'  # Return nibabel objects
)

simulator = DosimetrySimulator(config)
results = simulator.run(
    ct_image=ct_img,
    activity_map=activity_img
)

# Access results
total_dose = results['total_dose']  # nibabel.Nifti1Image
uncertainty = results['uncertainty']
metrics = results['metrics']  # Performance data

print(f"Total dose range: {total_dose.get_data().min():.2f} - {total_dose.get_data().max():.2f} Gy")
```

### Example 3: Multiple Radionuclides

```python
from MCGPURPTDosimetry import DosimetrySimulator, SimulationConfig

# Simulate mixed Lu-177 / Y-90 injection
config = SimulationConfig(
    radionuclide='Lu-177',  # Simulate Lu-177 and daughters
    num_primaries=1_000_000,
    output_format='file',
    output_path='./results/'
)

simulator = DosimetrySimulator(config)
results = simulator.run(
    ct_image='patient_ct.nii.gz',
    activity_map='patient_activity.nii.gz'
)

# Outputs include:
# - Lu-177_dose.nii.gz (from Lu-177 decay)
# - Hf-177_dose.nii.gz (from Lu-177 → Hf-177)
# - total_dose.nii.gz (Lu-177 + Hf-177)
```

## Configuration Options

### Basic Configuration

```python
config = SimulationConfig(
    # Radionuclide to simulate
    radionuclide='Lu-177',          # Required: one of 25 supported
    
    # Simulation parameters
    num_primaries=1_000_000,        # Number of particle histories
    energy_cutoff_keV=10.0,         # Min particle energy (default: 10 keV)
    num_batches=10,                 # Batches for uncertainty (default: 10)
    
    # I/O
    output_format='file',           # 'file' or 'object'
    output_path='./results/',       # Directory for output files
    
    # Hardware
    device='cuda',                  # 'cuda' (GPU) or 'cpu'
    
    # Optional
    random_seed=42                  # For reproducibility
)
```

### Advanced Configuration

```python
# For production quality (high accuracy)
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=10_000_000,       # 10 million primaries
    energy_cutoff_keV=1.0,          # Low energy cutoff
    num_batches=20,                 # More batches for better uncertainty
    device='cuda'
)

# For rapid testing (low accuracy)
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=10_000,           # Only 10k primaries
    energy_cutoff_keV=50.0,         # Higher cutoff (faster)
    num_batches=5,
    device='cuda'
)

# Custom HU-to-material mapping
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=1_000_000,
    hu_to_material_lut={
        "Air": [(-1000, -950)],
        "Lung": [(-950, -500)],
        "Soft_Tissue": [(-80, 100), (100, 300)],  # Multi-range for contrast
        "Bone": [(300, 1500)]
    }
)
```

## Physics Configuration

### Monte Carlo Engine Options

```python
from MCGPURPTDosimetry import DosimetrySimulator, SimulationConfig

config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=1_000_000,
)

simulator = DosimetrySimulator(config)

# Optional: configure physics
mc_config = {
    'enable_bremsstrahlung': True,   # Photon generation (default: True)
    'enable_delta_rays': False,      # Knock-on electrons (default: False)
    'enable_fluorescence': True,     # X-rays from photoelectric (default: True)
    'max_particles_in_flight': 100000  # GPU memory limit
}

results = simulator.run(
    ct_image='patient_ct.nii.gz',
    activity_map='patient_activity.nii.gz',
    # Note: Physics config passed via SimulationConfig in actual API
)
```

## Using Segmentation Masks

For organ-specific dose calculation or contrast-enhanced CT handling:

```python
from MCGPURPTDosimetry import DosimetrySimulator, SimulationConfig

config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=1_000_000,
)

simulator = DosimetrySimulator(config)

# Define tissues using segmentation masks
tissue_masks = {
    'Tumor': 'tumor_mask.nii.gz',        # Binary or label mask
    'Liver': 'liver_mask.nii.gz',
    'Kidney': 'kidney_mask.nii.gz',
}

results = simulator.run(
    ct_image='patient_ct.nii.gz',
    activity_map='patient_activity.nii.gz',
    tissue_masks=tissue_masks,
    mask_priority_order=['Tumor', 'Liver', 'Kidney'],  # Overlap priority
    use_ct_density=True  # Use CT-derived density in masked regions
)

# Dose calculation respects mask-based tissue assignment
```

## Supported Radionuclides

### Therapeutic (10)
- **Lu-177**: Peptide receptor therapy, T₁/₂ = 6.75 days
- **Y-90**: Liver cancer (microspheres), T₁/₂ = 2.67 days
- **I-131**: Thyroid cancer, T₁/₂ = 8.03 days
- **Re-188**: Radiopharmaceutical therapy, T₁/₂ = 17 hours
- **Cu-67**: Diagnostic/therapeutic, T₁/₂ = 2.57 days
- **Ho-166**: Therapeutic, T₁/₂ = 26.8 hours
- **Tb-161**: Therapeutic, T₁/₂ = 6.89 days
- **At-211**: Targeted alpha therapy, T₁/₂ = 7.2 hours
- **Ac-225**: Targeted alpha therapy (decay chain), T₁/₂ = 10 days
- **Pb-212**: Decay chain product, T₁/₂ = 10.6 hours

### Diagnostic (8)
- **Tc-99m**: SPECT imaging (most common), T₁/₂ = 6.01 hours
- **F-18**: PET imaging (FDG-PET), T₁/₂ = 110 minutes
- **Ga-68**: PET imaging, T₁/₂ = 68 minutes
- **Cu-64**: Dual PET/SPECT, T₁/₂ = 12.7 hours
- **C-11**: PET imaging, T₁/₂ = 20.4 minutes
- **N-13**: PET imaging, T₁/₂ = 10 minutes
- **Zr-89**: PET imaging (antibodies), T₁/₂ = 3.27 days
- **I-124**: PET imaging (iodine), T₁/₂ = 4.18 days

See [RADIONUCLIDE_DATABASE.md](RADIONUCLIDE_DATABASE.md) for complete decay data.

## Output Files

### File-Based Output (`output_format='file'`)

The simulation generates NIfTI files in the output directory:

```
./results/
├── total_dose.nii.gz              # Total dose (sum of all nuclides)
├── Lu-177_dose.nii.gz             # Individual nuclide dose
├── Hf-177_dose.nii.gz             # Daughter nuclide dose
├── uncertainty.nii.gz             # Per-voxel uncertainty (%)
└── simulation.log                 # Simulation output and metrics
```

### Object-Based Output (`output_format='object'`)

Returns Python dictionary with nibabel objects:

```python
results = {
    'total_dose': nibabel.Nifti1Image,     # Total dose
    'individual_doses': {                   # Per-nuclide doses
        'Lu-177': nibabel.Nifti1Image,
        'Hf-177': nibabel.Nifti1Image,
    },
    'uncertainty': nibabel.Nifti1Image,    # Uncertainty map
    'metrics': {                            # Performance metrics
        'total_time_sec': 120.5,
        'primaries_per_second': 8322.0,
        'gpu_memory_mb': 2048.0
    }
}
```

## Performance Tips

### For Maximum Speed
```python
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=100_000,         # Fewer primaries
    energy_cutoff_keV=100.0,       # Higher cutoff
    num_batches=1,                 # Single batch
    device='cuda'
)
```

### For Maximum Accuracy
```python
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=10_000_000,      # Many primaries
    energy_cutoff_keV=1.0,         # Low cutoff
    num_batches=20,                # Multiple batches
    device='cuda'
)
```

### Memory Management

On GPU with limited memory:

```python
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=1_000_000,
)
simulator = DosimetrySimulator(config)

# Can also configure in MC engine:
# max_particles_in_flight: 50000 (reduce from default 100000)
# This trades latency for memory usage
```

## Troubleshooting

### CUDA Out of Memory
```python
# Reduce particle batch size
config = SimulationConfig(
    radionuclide='Lu-177',
    num_primaries=500_000,  # Reduce primaries
)

# Or use CPU
config = SimulationConfig(
    radionuclide='Lu-177',
    device='cpu'            # Fall back to CPU
)
```

### Slow Performance
```python
# Check GPU availability
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Your GPU name

# Verify using GPU
config = SimulationConfig(
    device='cuda'
)

# Monitor GPU usage (in separate terminal)
nvidia-smi --loop=1  # Real-time GPU stats
```

### File I/O Issues

```python
from pathlib import Path

# Create output directory
output_dir = Path('./results')
output_dir.mkdir(parents=True, exist_ok=True)

config = SimulationConfig(
    output_format='file',
    output_path=str(output_dir)  # Use absolute path
)
```

## Running Examples

Included examples demonstrate common workflows:

```bash
# Basic usage example
python examples/basic_usage.py

# Full simulation with decay chains
python examples/full_simulation_example.py

# Segmentation mask workflow
python examples/mask_based_tissue_definition.py

# Physics features demonstration
python examples/physics_demonstration.py
```

## Next Steps

- **For physics details**: See [PHYSICS.md](PHYSICS.md)
- **For database information**: See [RADIONUCLIDE_DATABASE.md](RADIONUCLIDE_DATABASE.md)
- **For architecture details**: See [IMPLEMENTATION_DETAILS.md](IMPLEMENTATION_DETAILS.md)
- **For API reference**: See [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
- **For mask workflow**: See [docs/MASK_BASED_WORKFLOW.md](docs/MASK_BASED_WORKFLOW.md)

