"""Generate minimal physics databases for testing."""

import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from MCGPURPTDosimetry.physics_data_preparation import (
    DecayDatabaseGenerator,
    CrossSectionGenerator
)


def generate_decay_database():
    """Generate minimal decay database."""
    print("Generating minimal decay database...")
    
    # Initialize generator
    generator = DecayDatabaseGenerator(icrp107_data_path='./icrp107_data')
    
    # Parse common nuclides
    nuclides = ['Lu-177', 'I-131', 'Y-90']
    generator.parse_icrp107(nuclides)
    
    # Generate database
    output_path = 'MCGPURPTDosimetry/physics_data/decay_databases/default.json'
    generator.generate_database(output_path)
    
    # Validate
    if generator.validate_database(output_path):
        print(f"✓ Decay database generated: {output_path}")
    else:
        print(f"✗ Decay database validation failed")
    
    return output_path


def generate_cross_section_database():
    """Generate minimal cross-section database."""
    print("\nGenerating minimal cross-section database...")
    
    # Initialize generator
    generator = CrossSectionGenerator(physics_backend='geant4')
    
    # Define materials
    materials = {
        'Water': {
            'composition': {'H': 0.111, 'O': 0.889},
            'density': 1.0
        },
        'Soft_Tissue': {
            'composition': {
                'H': 0.102, 'C': 0.143, 'N': 0.034, 'O': 0.708,
                'Na': 0.002, 'P': 0.003, 'S': 0.003, 'Cl': 0.002, 'K': 0.003
            },
            'density': 1.04
        },
        'Bone': {
            'composition': {
                'H': 0.034, 'C': 0.155, 'N': 0.042, 'O': 0.435,
                'Mg': 0.002, 'P': 0.103, 'S': 0.003, 'Ca': 0.226
            },
            'density': 1.92
        },
        'Air': {
            'composition': {'N': 0.755, 'O': 0.232, 'Ar': 0.013},
            'density': 0.001205
        }
    }
    
    for name, props in materials.items():
        generator.define_material(name, props['composition'], props['density'])
    
    # Create energy grid (10 eV to 10 MeV, 1000 points)
    energy_grid = np.logspace(1, 7, 1000)  # eV
    
    # Calculate cross-sections
    generator.calculate_cross_sections(energy_grid, list(materials.keys()))
    
    # Export database
    output_path = 'MCGPURPTDosimetry/physics_data/cross_section_databases/default.h5'
    generator.export_database(output_path)
    
    print(f"✓ Cross-section database generated: {output_path}")
    
    return output_path


def main():
    """Generate all minimal databases."""
    print("=" * 60)
    print("Generating Minimal Physics Databases")
    print("=" * 60)
    
    # Generate databases
    decay_db_path = generate_decay_database()
    xs_db_path = generate_cross_section_database()
    
    print("\n" + "=" * 60)
    print("Database Generation Complete")
    print("=" * 60)
    print(f"\nDecay database: {decay_db_path}")
    print(f"Cross-section database: {xs_db_path}")
    print("\nThese databases can be used for testing and development.")
    print("For production use, generate databases from actual ICRP-107")
    print("data and validated physics libraries (Geant4/PENELOPE).")


if __name__ == '__main__':
    main()
