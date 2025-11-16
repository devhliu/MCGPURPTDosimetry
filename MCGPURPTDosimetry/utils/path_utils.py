"""Path validation utilities for secure file operations."""

from pathlib import Path
from typing import Union
import os


class PathValidationError(Exception):
    """Raised when path validation fails."""
    pass


def validate_path(path: Union[str, Path], must_exist: bool = False) -> Path:
    """Validate and sanitize a file path.
    
    Args:
        path: Path to validate
        must_exist: If True, path must exist
        
    Returns:
        Validated Path object
        
    Raises:
        PathValidationError: If path is invalid or unsafe
    """
    try:
        path_obj = Path(path).resolve()
    except (ValueError, OSError) as e:
        raise PathValidationError(f"Invalid path: {path}") from e
    
    # Check for directory traversal attempts
    try:
        # Ensure the resolved path doesn't escape expected boundaries
        # This is a basic check; adjust based on your security requirements
        if '..' in path_obj.parts:
            raise PathValidationError(f"Path contains directory traversal: {path}")
    except Exception as e:
        raise PathValidationError(f"Path validation failed: {path}") from e
    
    # Check existence if required
    if must_exist and not path_obj.exists():
        raise PathValidationError(f"Path does not exist: {path}")
    
    return path_obj


def validate_output_path(path: Union[str, Path], create_parents: bool = True) -> Path:
    """Validate and prepare an output path.
    
    Args:
        path: Output path to validate
        create_parents: If True, create parent directories
        
    Returns:
        Validated Path object
        
    Raises:
        PathValidationError: If path is invalid
    """
    path_obj = validate_path(path, must_exist=False)
    
    # Create parent directories if requested
    if create_parents:
        try:
            path_obj.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise PathValidationError(
                f"Cannot create parent directories for: {path}"
            ) from e
    
    return path_obj


def is_safe_path(path: Union[str, Path], base_dir: Union[str, Path] = None) -> bool:
    """Check if a path is safe (no directory traversal).
    
    Args:
        path: Path to check
        base_dir: Optional base directory to restrict to
        
    Returns:
        True if path is safe, False otherwise
    """
    try:
        path_obj = Path(path).resolve()
        
        if base_dir is not None:
            base_obj = Path(base_dir).resolve()
            # Check if path is within base directory
            try:
                path_obj.relative_to(base_obj)
            except ValueError:
                return False
        
        # Check for suspicious patterns
        if '..' in path_obj.parts:
            return False
        
        return True
    except Exception:
        return False
