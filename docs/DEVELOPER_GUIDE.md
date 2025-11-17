# Developer Guide

## Overview

This guide is for developers who want to contribute to, extend, or modify MCGPURPTDosimetry. It covers development setup, coding standards, testing procedures, and contribution workflows.

## Development Environment Setup

### Prerequisites

- Python 3.8+ with pip
- Git for version control
- CUDA Toolkit (optional, for GPU development)
- Text editor or IDE (VS Code, PyCharm recommended)

### Initial Setup

```bash
# Clone repository
git clone https://github.com/devhliu/MCGPURPTDosimetry.git
cd MCGPURPTDosimetry

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy

# Install package in editable mode
pip install -e .

# Generate physics databases
python scripts/generate_minimal_databases.py
```

### IDE Configuration

#### VS Code

Create `.vscode/settings.json`:

```json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "100"],
    "editor.formatOnSave": true,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"]
}
```

#### PyCharm

1. Set Python interpreter to virtual environment
2. Enable pytest as test runner
3. Configure Black as code formatter
4. Enable flake8 for linting

## Coding Standards

### Style Guide

Follow PEP 8 with these modifications:

- **Line length**: 100 characters (not 79)
- **Docstring style**: Google style
- **Type hints**: Required for public APIs
- **Imports**: Organized with isort

### Code Formatting

Use Black for automatic formatting:

```bash
# Format all Python files
black MCGPURPTDosimetry/ tests/ scripts/

# Check formatting without changes
black --check MCGPURPTDosimetry/
```

### Linting

Use flake8 for style checking:

```bash
# Run flake8
flake8 MCGPURPTDosimetry/ tests/ scripts/

# Configuration in setup.cfg
[flake8]
max-line-length = 100
exclude = __pycache__,.git,build,dist
ignore = E203,W503  # Black compatibility
```

### Type Checking

Use mypy for static type checking:

```bash
# Run mypy
mypy MCGPURPTDosimetry/

# Configuration in setup.cfg
[mypy]
python_version = 3.8
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
```

### Documentation Standards

#### Docstring Format

Use Google-style docstrings:

```python
def calculate_dose(
    energy: torch.Tensor,
    position: torch.Tensor,
    material: torch.Tensor
) -> torch.Tensor:
    """Calculate dose deposition for particles.
    
    This function computes the energy deposited per unit mass for
    particles at given positions in specified materials.
    
    Args:
        energy: Particle energies in keV [N]
        position: Particle positions in voxel coordinates [N, 3]
        material: Material IDs at positions [N]
        
    Returns:
        Dose in Gy for each particle [N]
        
    Raises:
        ValueError: If energy contains negative values
        RuntimeError: If material IDs are out of range
        
    Example:
        >>> energy = torch.tensor([100.0, 200.0, 300.0])
        >>> position = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        >>> material = torch.tensor([1, 1, 2])
        >>> dose = calculate_dose(energy, position, material)
    """
    pass
```

#### Module Documentation

Every module should have a docstring:

```python
"""Photon interaction physics module.

This module implements photon transport physics including:
- Photoelectric effect with characteristic X-rays
- Compton scattering (Klein-Nishina)
- Pair production
- Rayleigh scattering

The implementation is optimized for GPU execution using PyTorch.
"""
```

## Project Structure

### Directory Organization

```
MCGPURPTDosimetry/
├── MCGPURPTDosimetry/          # Main package
│   ├── core/                   # Core simulation components
│   ├── physics/                # Physics engines
│   ├── utils/                  # Utilities
│   └── physics_data/           # Data files
├── tests/                      # Test suite
│   ├── test_core/              # Core module tests
│   ├── test_physics/           # Physics module tests
│   └── test_integration/       # Integration tests
├── scripts/                    # Utility scripts
│   ├── generate_*.py           # Database generation
│   └── validate_*.py           # Validation scripts
├── examples/                   # Usage examples
├── docs/                       # Documentation
├── requirements.txt            # Dependencies
├── setup.py                    # Package setup
└── README.md                   # Main documentation
```

### Adding New Modules

1. Create module file in appropriate directory
2. Add `__init__.py` if creating new package
3. Import in parent `__init__.py`
4. Add tests in corresponding test directory
5. Update documentation

## Testing

### Test Organization

```
tests/
├── test_core/
│   ├── test_dosimetry_simulator.py
│   ├── test_input_manager.py
│   ├── test_geometry_processor.py
│   └── test_source_term_processor.py
├── test_physics/
│   ├── test_monte_carlo_engine.py
│   ├── test_photon_physics.py
│   ├── test_electron_physics.py
│   └── test_beta_spectrum.py
├── test_integration/
│   ├── test_full_simulation.py
│   └── test_phantom_cases.py
└── conftest.py  # Shared fixtures
```

### Writing Tests

#### Unit Test Example

```python
import pytest
import torch
from MCGPURPTDosimetry.core import GeometryProcessor

class TestGeometryProcessor:
    """Test suite for GeometryProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create GeometryProcessor instance."""
        return GeometryProcessor()
    
    @pytest.fixture
    def sample_ct(self):
        """Create sample CT data."""
        return torch.randn(64, 64, 64) * 500  # HU values
    
    def test_create_geometry_data(self, processor, sample_ct):
        """Test geometry data creation."""
        voxel_size = (2.0, 2.0, 2.0)
        affine = torch.eye(4)
        
        geometry = processor.create_geometry_data(
            sample_ct, voxel_size, affine
        )
        
        assert geometry.material_map.shape == (64, 64, 64)
        assert geometry.density_map.shape == (64, 64, 64)
        assert geometry.voxel_size == voxel_size
    
    def test_hu_to_material_mapping(self, processor):
        """Test HU to material conversion."""
        ct = torch.tensor([[-1000, -500, 0, 500, 1000]])
        
        material_map = processor._map_hu_to_materials(ct)
        
        # Check material assignments
        assert material_map[0, 0] == 0  # Air
        assert material_map[0, 1] == 1  # Lung
        assert material_map[0, 2] == 3  # Soft tissue
        assert material_map[0, 3] == 5  # Bone
```

#### Integration Test Example

```python
import pytest
from MCGPURPTDosimetry import DosimetrySimulator, SimulationConfig

class TestFullSimulation:
    """Integration tests for complete simulation workflow."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SimulationConfig(
            radionuclide='Lu-177',
            num_primaries=1000,
            output_format='object',
            device='cpu'  # Use CPU for CI
        )
    
    def test_small_phantom_simulation(self, config, tmp_path):
        """Test simulation on small phantom."""
        # Create synthetic phantom
        ct_image = create_water_phantom(shape=(32, 32, 32))
        activity_map = create_point_source(shape=(32, 32, 32))
        
        # Run simulation
        simulator = DosimetrySimulator(config)
        results = simulator.run(ct_image, activity_map)
        
        # Verify results
        assert 'total_dose' in results
        assert results['total_dose'].shape == (32, 32, 32)
        assert results['total_dose'].max() > 0
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_core/test_geometry_processor.py

# Run specific test
pytest tests/test_core/test_geometry_processor.py::TestGeometryProcessor::test_create_geometry_data

# Run with coverage
pytest --cov=MCGPURPTDosimetry --cov-report=html

# Run only fast tests (skip slow integration tests)
pytest -m "not slow"

# Run in parallel
pytest -n auto
```

### Test Fixtures

Create reusable fixtures in `conftest.py`:

```python
import pytest
import torch
import nibabel as nib

@pytest.fixture
def device():
    """Get computation device."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'

@pytest.fixture
def water_phantom():
    """Create water phantom."""
    data = torch.zeros(64, 64, 64)  # 0 HU = water
    affine = torch.eye(4).numpy()
    return nib.Nifti1Image(data.numpy(), affine)

@pytest.fixture
def uniform_activity():
    """Create uniform activity distribution."""
    data = torch.ones(64, 64, 64) * 1e6  # 1 MBq/voxel
    affine = torch.eye(4).numpy()
    return nib.Nifti1Image(data.numpy(), affine)
```

## Debugging

### Logging

Use the logging system for debugging:

```python
from MCGPURPTDosimetry.utils.logging import get_logger

logger = get_logger()

logger.debug("Detailed debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error message")
```

### GPU Debugging

#### Check GPU Usage

```python
import torch

# Check if GPU is available
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")

# Monitor memory
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
```

#### Profile GPU Code

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
) as prof:
    # Run simulation
    results = simulator.run(ct_image, activity_map)

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### Common Issues

#### Out of Memory

```python
# Reduce particle batch size
config = SimulationConfig(
    max_particles_in_flight=50000  # Reduce from default 100000
)

# Clear cache between simulations
torch.cuda.empty_cache()
```

#### Slow Performance

```python
# Profile to find bottlenecks
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run code
results = simulator.run(ct_image, activity_map)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

## Contributing

### Contribution Workflow

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a feature branch**: `git checkout -b feature/my-feature`
4. **Make changes** and commit: `git commit -m "Add my feature"`
5. **Write tests** for new functionality
6. **Run tests** to ensure nothing breaks
7. **Push to your fork**: `git push origin feature/my-feature`
8. **Create pull request** on GitHub

### Commit Message Guidelines

Follow conventional commits:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Example**:
```
feat(physics): Add Rayleigh scattering model

Implement coherent scattering for photons using form factor
approximation. This improves accuracy for low-energy photons
in high-Z materials.

Closes #123
```

### Pull Request Guidelines

**Before submitting**:
- [ ] Code follows style guidelines (Black, flake8)
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages follow guidelines

**PR Description Template**:
```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Changes
- List of specific changes
- Another change

## Testing
How was this tested?

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Code formatted with Black
```

## Adding New Features

### Adding a New Radionuclide

1. **Add decay data** to `physics_data/decay_databases/default.json`:

```json
{
  "My-Nuclide": {
    "atomic_number": 50,
    "mass_number": 100,
    "half_life_seconds": 86400.0,
    "decay_modes": {
      "beta_minus": {
        "branching_ratio": 1.0,
        "daughter": "Daughter-Nuclide",
        "emissions": [
          {
            "type": "beta_minus",
            "energy_keV": 200.0,
            "max_energy_keV": 500.0,
            "intensity": 1.0
          }
        ]
      }
    }
  }
}
```

2. **Regenerate databases**:
```bash
python scripts/generate_minimal_databases.py
```

3. **Test**:
```python
config = SimulationConfig(radionuclide='My-Nuclide', ...)
```

### Adding a New Physics Model

1. **Create module** in `MCGPURPTDosimetry/physics/`:

```python
# my_physics.py
import torch

class MyPhysicsModel:
    """New physics model."""
    
    def __init__(self, config):
        self.config = config
    
    def simulate_interaction(self, particles):
        """Simulate new interaction type."""
        # Implementation
        pass
```

2. **Integrate** in `MonteCarloEngine`:

```python
from .my_physics import MyPhysicsModel

class MonteCarloEngine:
    def __init__(self, ...):
        # ...
        self.my_physics = MyPhysicsModel(config)
    
    def _transport_particles(self, particles):
        # ...
        if enable_my_physics:
            self.my_physics.simulate_interaction(particles)
```

3. **Add tests**:

```python
# tests/test_physics/test_my_physics.py
def test_my_physics_model():
    model = MyPhysicsModel(config)
    result = model.simulate_interaction(particles)
    assert result is not None
```

### Adding a New Output Format

1. **Extend** `DoseSynthesis.export_results()`:

```python
def export_results(self, output_format, output_path):
    if output_format == 'my_format':
        return self._export_my_format(output_path)
    # ... existing formats
    
def _export_my_format(self, output_path):
    """Export in custom format."""
    # Implementation
    pass
```

2. **Update** configuration validation:

```python
# In SimulationConfig._validate()
if self.output_format not in ['file', 'object', 'my_format']:
    raise ValueError(f"Invalid output_format: {self.output_format}")
```

## Performance Optimization

### Profiling

Use PyTorch profiler:

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    results = simulator.run(ct_image, activity_map)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Optimization Checklist

- [ ] Use vectorized operations instead of loops
- [ ] Preallocate tensors when possible
- [ ] Use in-place operations (`+=`, `-=`)
- [ ] Minimize CPU-GPU data transfers
- [ ] Batch operations for better GPU utilization
- [ ] Use appropriate data types (float32 vs float64)
- [ ] Profile before and after optimization

### Memory Optimization

```python
# Use torch.no_grad() for inference
with torch.no_grad():
    results = model(inputs)

# Delete unused tensors
del large_tensor
torch.cuda.empty_cache()

# Use gradient checkpointing for large models
torch.utils.checkpoint.checkpoint(function, *args)
```

## Documentation

### Building Documentation

```bash
# Install Sphinx
pip install sphinx sphinx-rtd-theme

# Build HTML documentation
cd docs
make html

# View documentation
open _build/html/index.html
```

### Writing Documentation

- Update relevant `.md` files in `docs/`
- Add docstrings to all public APIs
- Include code examples
- Update API reference if adding new classes/functions

## Release Process

### Version Numbering

Follow semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Incompatible API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Checklist

1. [ ] Update version in `setup.py` and `__init__.py`
2. [ ] Update CHANGELOG.md
3. [ ] Run full test suite
4. [ ] Update documentation
5. [ ] Create git tag: `git tag v0.2.0`
6. [ ] Push tag: `git push origin v0.2.0`
7. [ ] Create GitHub release
8. [ ] Build and upload to PyPI (if applicable)

## Getting Help

### Resources

- **Documentation**: Read the docs in `docs/`
- **Examples**: Check `examples/` directory
- **Tests**: Look at test files for usage patterns
- **Issues**: Search GitHub issues for similar problems

### Asking Questions

When asking for help, provide:

1. **Environment**: OS, Python version, GPU model
2. **Code**: Minimal reproducible example
3. **Error**: Full error message and traceback
4. **Context**: What you're trying to achieve

### Reporting Bugs

Include in bug reports:

1. **Description**: What happened vs what you expected
2. **Reproduction**: Steps to reproduce the bug
3. **Environment**: System information
4. **Logs**: Relevant log output
5. **Data**: Sample data if applicable (anonymized)

---

**Next**: See [API_REFERENCE.md](API_REFERENCE.md) for detailed API documentation.
