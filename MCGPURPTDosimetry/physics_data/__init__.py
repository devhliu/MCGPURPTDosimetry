"""Physics data package containing decay and cross-section databases.

This package contains pre-computed physics databases that are bundled
with the MCGPURPTDosimetry package for immediate use.
"""

import os
from pathlib import Path


def get_physics_data_dir() -> Path:
    """Get the physics data directory path.
    
    Returns:
        Path to the physics_data directory
    """
    return Path(__file__).parent


def get_decay_database_path(database_name: str = 'default.json') -> Path:
    """Get path to a decay database file.
    
    Args:
        database_name: Name of the database file (default: 'default.json')
        
    Returns:
        Path to the decay database file
        
    Raises:
        FileNotFoundError: If the database file doesn't exist
    """
    db_path = get_physics_data_dir() / 'decay_databases' / database_name
    if not db_path.exists():
        raise FileNotFoundError(
            f"Decay database not found: {db_path}\n"
            f"Available databases: {list_decay_databases()}"
        )
    return db_path


def get_cross_section_database_path(database_name: str = 'default.h5') -> Path:
    """Get path to a cross-section database file.
    
    Args:
        database_name: Name of the database file (default: 'default.h5')
        
    Returns:
        Path to the cross-section database file
        
    Raises:
        FileNotFoundError: If the database file doesn't exist
    """
    db_path = get_physics_data_dir() / 'cross_section_databases' / database_name
    if not db_path.exists():
        raise FileNotFoundError(
            f"Cross-section database not found: {db_path}\n"
            f"Available databases: {list_cross_section_databases()}"
        )
    return db_path


def list_decay_databases() -> list:
    """List all available decay databases.
    
    Returns:
        List of decay database filenames
    """
    decay_dir = get_physics_data_dir() / 'decay_databases'
    if not decay_dir.exists():
        return []
    return [f.name for f in decay_dir.glob('*.json')]


def list_cross_section_databases() -> list:
    """List all available cross-section databases.
    
    Returns:
        List of cross-section database filenames
    """
    xs_dir = get_physics_data_dir() / 'cross_section_databases'
    if not xs_dir.exists():
        return []
    return [f.name for f in xs_dir.glob('*.h5')]


# Default database paths (with safe initialization)
try:
    DEFAULT_DECAY_DATABASE = str(get_decay_database_path())
except FileNotFoundError:
    DEFAULT_DECAY_DATABASE = None

try:
    DEFAULT_CROSS_SECTION_DATABASE = str(get_cross_section_database_path())
except FileNotFoundError:
    DEFAULT_CROSS_SECTION_DATABASE = None


__all__ = [
    'get_physics_data_dir',
    'get_decay_database_path',
    'get_cross_section_database_path',
    'list_decay_databases',
    'list_cross_section_databases',
    'DEFAULT_DECAY_DATABASE',
    'DEFAULT_CROSS_SECTION_DATABASE',
]
