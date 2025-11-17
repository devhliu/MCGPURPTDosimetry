# MCGPURPTDosimetry Documentation

Welcome to the MCGPURPTDosimetry documentation. This guide will help you understand and use the GPU-accelerated Monte Carlo dosimetry system for radiopharmaceutical therapy.

## Documentation Structure

The documentation is organized into six comprehensive guides covering different aspects of the system:

### 1. **USER_GUIDE.md** - Getting Started for Users
Start here if you want to **use** MCGPURPTDosimetry for dosimetry calculations.

**Contains**:
- Installation instructions
- Quick start examples
- Basic and advanced usage
- Physics data preparation
- Performance optimization
- Troubleshooting guide

**When to read**: Before running your first simulation or when looking for usage examples.

### 2. **DEVELOPER_GUIDE.md** - For Contributors
Start here if you want to **develop** or **contribute** to MCGPURPTDosimetry.

**Contains**:
- Development environment setup
- Coding standards (PEP 8, docstrings, type hints)
- Testing procedures and examples
- Debugging techniques
- Contributing workflow and guidelines
- Adding new features (radionuclides, physics models, output formats)
- Release process

**When to read**: Before making code changes or contributing to the project.

### 3. **API_REFERENCE.md** - Complete API Documentation
Detailed reference for all public classes, methods, and functions.

**Contains**:
- Core classes (DosimetrySimulator, SimulationConfig)
- Physics classes (DecayDatabase, CrossSectionDatabase)
- Data preparation classes (DecayDatabaseGenerator, CrossSectionGenerator)
- Utility classes (InputManager, GeometryProcessor)
- Data models (GeometryData, NuclideData, MaterialProperties)
- Available radionuclides and materials
- Error handling and exceptions
- Performance tips

**When to read**: When you need to understand specific classes, methods, or parameters.

### 4. **DESIGN_ARCHITECTURE.md** - System Design & Architecture
Technical documentation of system architecture and implementation details.

**Contains**:
- Architectural principles and patterns
- High-level system architecture and data flow
- Module descriptions and responsibilities
- Design patterns used throughout the codebase
- GPU programming patterns and optimizations
- Error handling strategy
- Performance optimization strategies
- Testing strategy
- Code organization and best practices
- Implementation details for all major components

**When to read**: When you need to understand how the system works internally or plan major changes.

### 5. **RADIONUCLIDE_PHYSICS.md** - Physics Database & Models
Comprehensive documentation of physics models and nuclear data.

**Contains**:
- Supported radionuclides (25 therapeutic and diagnostic)
- Decay data structure and sources
- Cross-section database (11 tissue materials)
- Interaction types (photon, electron, alpha)
- Physics models (photon transport, electron transport, alpha handling)
- Data generation process
- Code integration details
- Accuracy and validation information
- Performance optimization
- Usage examples
- References and citations

**When to read**: When you need physics details or want to customize the nuclear data.

### 6. **SOFTWARE_FEATURES.md** - Features & Capabilities
Overview of all features and capabilities of MCGPURPTDosimetry.

**Contains**:
- GPU acceleration details
- Production-grade physics validation
- Multi-particle transport
- Decay chain support
- Flexible input/output formats
- Uncertainty quantification
- Contrast-enhanced CT support
- Segmentation mask support
- Comprehensive radionuclide database
- Material database
- Beta spectrum sampling
- Configurable physics
- Performance metrics
- Comprehensive logging
- Dose statistics
- Advanced features (YAML config, reproducibility, memory management)
- Validation tools
- Feature comparison with other codes
- Future features and research directions

**When to read**: To understand what MCGPURPTDosimetry can do and how it compares to other codes.

## Additional Resources

### Supporting Documentation

- **MASK_BASED_WORKFLOW.md** - Detailed guide for using segmentation masks in dosimetry calculations
- **examples/** - Working code examples demonstrating common workflows
- **scripts/** - Utility scripts for database generation and validation
- **tests/** - Test suite showing usage patterns and expected behavior

### Quick Navigation

**For Different User Types:**

- **Clinical User**: Start with USER_GUIDE.md → Run examples → API_REFERENCE.md as needed
- **Researcher**: USER_GUIDE.md → RADIONUCLIDE_PHYSICS.md → SOFTWARE_FEATURES.md → Examples
- **Developer**: DEVELOPER_GUIDE.md → DESIGN_ARCHITECTURE.md → API_REFERENCE.md → tests/
- **Algorithm Expert**: DESIGN_ARCHITECTURE.md → RADIONUCLIDE_PHYSICS.md → SOFTWARE_FEATURES.md

**By Task:**

- **Installation**: See USER_GUIDE.md → Installation section
- **First simulation**: USER_GUIDE.md → Quick Start → Run examples
- **Optimize performance**: USER_GUIDE.md → Performance Optimization
- **Understand design**: DESIGN_ARCHITECTURE.md
- **Look up API**: API_REFERENCE.md
- **Understand physics**: RADIONUCLIDE_PHYSICS.md
- **Add feature**: DEVELOPER_GUIDE.md → Adding New Features
- **Debug issue**: USER_GUIDE.md → Troubleshooting → DEVELOPER_GUIDE.md → Debugging
- **Use segmentation masks**: MASK_BASED_WORKFLOW.md
- **See code example**: examples/ directory

## Documentation Quality

All documentation is maintained to match the actual code:

- ✅ Code examples are tested and working
- ✅ Class and method signatures are accurate
- ✅ Available radionuclides and materials are current
- ✅ Performance numbers are validated
- ✅ Error messages match actual implementation
- ✅ API documentation automatically references source code

## Getting Help

1. **Check the relevant guide** based on your task (use Quick Navigation above)
2. **Search the documentation** for keywords related to your question
3. **Review the examples** in the `examples/` directory
4. **Check API_REFERENCE.md** for specific class/method details
5. **Enable debug logging** (see DEVELOPER_GUIDE.md → Logging)
6. **Open an issue** on the project repository with:
   - Your code or command that shows the problem
   - Full error message and traceback
   - Your system information (OS, Python, PyTorch, CUDA versions)
   - Expected vs actual behavior

## Key Features Overview

MCGPURPTDosimetry provides:

- **GPU Acceleration**: 10-20x speedup over CPU implementations
- **Production-Grade Physics**: ±5-10% accuracy validated against Geant4/PENELOPE
- **25 Radionuclides**: Therapeutic and diagnostic, with decay chains
- **11 Material Models**: Complete tissue properties and cross-sections
- **Flexible I/O**: File-based and in-memory workflows
- **Segmentation Masks**: Organ-specific tissue assignment
- **Uncertainty Quantification**: Per-voxel statistical uncertainty
- **Python API**: Easy integration into research pipelines

## Version

This documentation is for **MCGPURPTDosimetry v0.1.0**

## License

Documentation is provided under the same license as the MCGPURPTDosimetry source code.

---

**Start with** [USER_GUIDE.md](USER_GUIDE.md) if you're new to MCGPURPTDosimetry.

For more information, see the [GitHub repository](https://github.com/devhliu/MCGPURPTDosimetry).
