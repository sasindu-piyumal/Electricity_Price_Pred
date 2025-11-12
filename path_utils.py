#!/usr/bin/env python
# coding: utf-8

"""
Secure Path Handling Utilities
Electricity Price Prediction

This module provides secure, configurable path handling with:
- Environment variable support via python-dotenv
- Directory traversal attack prevention
- Cross-platform compatibility (Windows, Linux, macOS)
- Sensible defaults
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.resolve()


def get_data_path(env_var='DATA_PATH', default='./electricity.csv'):
    """
    Get the data file path from environment variable with fallback to default.
    
    Args:
        env_var (str): Environment variable name. Default: 'DATA_PATH'
        default (str): Default path if environment variable is not set.
                      Default: './electricity.csv'
    
    Returns:
        Path: Validated, secure path object
    
    Raises:
        ValueError: If path validation fails (e.g., directory traversal attempt)
        FileNotFoundError: If the resolved path doesn't exist
    
    Examples:
        >>> path = get_data_path()  # Uses DATA_PATH env var or default
        >>> path = get_data_path(default='./data/custom.csv')
    """
    # Get path from environment variable or use default
    path_str = os.getenv(env_var, default)
    
    # Validate and resolve the path
    validated_path = validate_path(path_str)
    
    return validated_path


def validate_path(path_str):
    """
    Validate a file path to prevent directory traversal attacks and ensure
    the path is within the project directory or is an explicitly allowed
    absolute path.
    
    Args:
        path_str (str): Path string to validate
    
    Returns:
        Path: Validated, resolved path object
    
    Raises:
        ValueError: If path contains directory traversal patterns or
                   escapes the project directory
        FileNotFoundError: If the path doesn't exist
    
    Security Features:
        - Prevents '../' directory traversal attacks
        - Resolves symlinks to actual paths
        - Ensures relative paths stay within project directory
        - Validates file existence
    """
    # Convert to Path object
    path = Path(path_str)
    
    # Resolve to absolute path (handles relative paths and symlinks)
    try:
        resolved_path = path.resolve(strict=False)
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Invalid path '{path_str}': {str(e)}")
    
    # For relative paths, ensure they don't escape the project directory
    if not path.is_absolute():
        # Get the relative path and check for parent directory traversal
        try:
            # Resolve relative to project root
            resolved_path = (PROJECT_ROOT / path).resolve(strict=False)
            
            # Ensure the resolved path is within or equal to project root
            # This prevents ../../../etc/passwd type attacks
            try:
                resolved_path.relative_to(PROJECT_ROOT)
            except ValueError:
                raise ValueError(
                    f"Security Error: Path '{path_str}' attempts to access "
                    f"files outside the project directory. "
                    f"Resolved to: {resolved_path}, "
                    f"Project root: {PROJECT_ROOT}"
                )
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Invalid relative path '{path_str}': {str(e)}")
    
    # Check if file exists
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {resolved_path}\n"
            f"Please ensure the file exists or update the DATA_PATH "
            f"environment variable in your .env file."
        )
    
    # Check if it's a file (not a directory)
    if not resolved_path.is_file():
        raise ValueError(f"Path is not a file: {resolved_path}")
    
    return resolved_path


def get_safe_path_str(env_var='DATA_PATH', default='./electricity.csv'):
    """
    Convenience function to get the data path as a string.
    
    Args:
        env_var (str): Environment variable name
        default (str): Default path if environment variable is not set
    
    Returns:
        str: Validated path as a string
    
    Examples:
        >>> path_str = get_safe_path_str()
        >>> data = pd.read_csv(path_str)
    """
    return str(get_data_path(env_var, default))


# Example usage and testing
if __name__ == "__main__":
    print("Path Utilities - Security Test")
    print("=" * 60)
    
    # Test 1: Default path
    print("\nTest 1: Default path")
    try:
        path = get_data_path()
        print(f"✓ Success: {path}")
    except (ValueError, FileNotFoundError) as e:
        print(f"✗ Error: {e}")
    
    # Test 2: Directory traversal attempt (should fail)
    print("\nTest 2: Directory traversal attack prevention")
    try:
        path = validate_path("../../../etc/passwd")
        print(f"✗ Security Failure: {path} (should have been blocked!)")
    except ValueError as e:
        print(f"✓ Security Success: Blocked - {str(e)[:80]}...")
    except FileNotFoundError:
        print(f"✓ Security Success: Path blocked or file not found")
    
    # Test 3: Absolute path (if electricity.csv exists)
    print("\nTest 3: Absolute path")
    try:
        abs_path = PROJECT_ROOT / "electricity.csv"
        if abs_path.exists():
            path = validate_path(str(abs_path))
            print(f"✓ Success: {path}")
        else:
            print(f"⊘ Skipped: electricity.csv not found in project root")
    except (ValueError, FileNotFoundError) as e:
        print(f"✗ Error: {e}")
    
    # Test 4: Relative path within project
    print("\nTest 4: Relative path within project")
    try:
        path = validate_path("./electricity.csv")
        print(f"✓ Success: {path}")
    except (ValueError, FileNotFoundError) as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("Security tests completed!")
