#!/usr/bin/env python3
"""Generate comprehensive cross-section database with all 9 tissue materials."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from MCGPURPTDosimetry.physics_data_preparation import CrossSectionGenerator


def main():
    """Generate comprehensive cross-section database."""
    print("=" * 60)
    print("Generating Comprehensive Cross-Section Database")
    print("=" * 60)
    
    # Initialize generator
    xs_gen = CrossSectionGenerator(physics_backend='geant4')
    
    # Define all 9 human body tissue materials based on ICRP reference data
    tissues = {
        'Air': {
            'composition': {'N': 0.755, 'O': 0.232, 'Ar': 0.013},
            'density': 0.0012
        },
        'Lung': {
            'composition': {
                'H': 0.103, 'C': 0.105, 'N': 0.031, 'O': 0.749,
                'P': 0.002, 'S': 0.002, 'Na': 0.002, 'Cl': 0.003, 'K': 0.002
            },
            'density': 0.26
        },
        'Muscle': {
            'composition': {
                'H': 0.102, 'C': 0.143, 'N': 0.034, 'O': 0.710,
                'P': 0.002, 'S': 0.003, 'Na': 0.001, 'Cl': 0.001, 'K': 0.004
            },
            'density': 1.05
        },
        'Soft_Tissue': {
            'composition': {
                'H': 0.105, 'C': 0.256, 'N': 0.027, 'O': 0.602,
                'P': 0.001, 'S': 0.003, 'Na': 0.001, 'Cl': 0.002, 'K': 0.003
            },
            'density': 1.04
        },
        'Fat': {
            'composition': {
                'H': 0.114, 'C': 0.598, 'N': 0.007, 'O': 0.278,
                'P': 0.001, 'S': 0.001, 'Na': 0.0005, 'Cl': 0.0005
            },
            'density': 0.95
        },
        'Bone_Cortical': {
            'composition': {
                'H': 0.034, 'C': 0.155, 'N': 0.042, 'O': 0.435,
                'P': 0.103, 'Ca': 0.225, 'Mg': 0.002, 'S': 0.003, 'Na': 0.001
            },
            'density': 1.92
        },
        'Bone_Trabecular': {
            'composition': {
                'H': 0.085, 'C': 0.404, 'N': 0.028, 'O': 0.367,
                'P': 0.034, 'Ca': 0.074, 'Mg': 0.002, 'S': 0.002, 'Na': 0.001, 'Cl': 0.002, 'K': 0.001
            },
            'density': 1.18
        },
        'Bone_Generic': {
            'composition': {
                'H': 0.064, 'C': 0.278, 'N': 0.027, 'O': 0.410,
                'P': 0.070, 'Ca': 0.147, 'Mg': 0.002, 'S': 0.002
            },
            'density': 1.55
        },
        'Water': {
            'composition': {'H': 0.111, 'O': 0.889},
            'density': 1.0
        },
        'Bone': {
            'composition': {
                'H': 0.064, 'C': 0.278, 'N': 0.027, 'O': 0.410,
                'P': 0.070, 'Ca': 0.147, 'Mg': 0.002, 'S': 0.002
            },
            'density': 1.85
        },
        'Iodine_Contrast_Mixture': {
            'composition': {
                'H': 0.100, 'C': 0.200, 'N': 0.025, 'O': 0.600,
                'I': 0.075
            },
            'density': 1.15
        }
    }
    
    print(f"\nDefining {len(tissues)} tissue materials...")
    for name, props in tissues.items():
        xs_gen.define_material(name, props['composition'], props['density'])
        print(f"  ✓ {name} (ρ={props['density']} g/cm³)")
    
    # Calculate cross-sections with high resolution at low energies
    print("\nCalculating cross-sections...")
    print("  Energy range: 10 eV to 10 MeV")
    print("  Energy points: 1000")
    
    energy_grid = np.logspace(1, 7, 1000)  # 10 eV to 10 MeV, 1000 points (in eV)
    xs_gen.calculate_cross_sections(energy_grid, list(tissues.keys()))
    
    # Export database
    output_path = 'MCGPURPTDosimetry/physics_data/cross_section_databases/default.h5'
    print(f"\nExporting database to: {output_path}")
    xs_gen.export_database(output_path)
    
    print("\n" + "=" * 60)
    print("Cross-section database generation complete!")
    print("=" * 60)
    print(f"\nDatabase contains {len(tissues)} materials:")
    for name in sorted(tissues.keys()):
        print(f"  - {name}")
    print()


if __name__ == '__main__':
    main()
