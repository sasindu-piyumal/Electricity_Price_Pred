"""
Configuration module for Electricity Data Analysis

This module provides configuration values and path validation utilities
for secure file operations with cross-platform compatibility.
"""

import os
from pathlib import Path
from typing import Union, Optional


# Project root directory - all paths are relative to this
PROJECT_ROOT = Path(__file__).parent.resolve()

# Data file path - configured relative to project root
DATA_FILE_PATH = PROJECT_ROOT / "electricity.csv"

# Allowed base directories for path traversal protection
ALLOWED_BASE_DIRECTORIES = [
    PROJECT_ROOT,
]


class PathValidationError(Exception):
    """Custom exception for path validation errors"""
    pass


def validate_file_path(
    file_path: Union[str, Path],
    allowed_extensions: Optional[list] = None,
    must_exist: bool = True,
    check_readable: bool = True
) -> Path:
    """
    Validate a file path with security checks and file existence validation.
    
    Args:
        file_path: Path to validate (string or Path object)
        allowed_extensions: List of allowed file extensions (e.g., ['.csv', '.txt'])
        must_exist: Whether the file must exist
        check_readable: Whether to check if file is readable
        
    Returns:
        Validated Path object (absolute path)
        
    Raises:
        PathValidationError: If validation fails
        
    Security:
        - Prevents path traversal attacks
        - Validates paths stay within allowed directories
        - Checks file extension if specified
        - Verifies file existence and readability
    """
    # Convert to Path object and resolve to absolute path
    try:
        file_path = Path(file_path).resolve()
    except (ValueError, OSError) as e:
        raise PathValidationError(
            f"Invalid path format: {file_path}\n"
            f"Error: {e}\n"
            f"Please ensure the path is valid and accessible."
        )
    
    # Check path traversal - ensure path is within allowed directories
    path_is_safe = False
    for allowed_base in ALLOWED_BASE_DIRECTORIES:
        try:
            # Check if file_path is relative to allowed_base
            file_path.relative_to(allowed_base)
            path_is_safe = True
            break
        except ValueError:
            continue
    
    if not path_is_safe:
        raise PathValidationError(
            f"Path traversal detected: {file_path}\n"
            f"File path must be within allowed directories:\n"
            f"{chr(10).join(str(d) for d in ALLOWED_BASE_DIRECTORIES)}\n"
            f"Please use a path within the project directory."
        )
    
    # Check file extension if specified
    if allowed_extensions:
        if file_path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
            raise PathValidationError(
                f"Invalid file extension: {file_path.suffix}\n"
                f"Allowed extensions: {', '.join(allowed_extensions)}\n"
                f"File: {file_path}"
            )
    
    # Check file existence
    if must_exist and not file_path.exists():
        raise PathValidationError(
            f"File not found: {file_path}\n"
            f"Please ensure:\n"
            f"1. The file exists at the specified location\n"
            f"2. The path is correct\n"
            f"3. You have permission to access the file"
        )
    
    # Check if it's actually a file (not a directory)
    if must_exist and not file_path.is_file():
        raise PathValidationError(
            f"Path is not a file: {file_path}\n"
            f"Expected a file, but found a directory or other type."
        )
    
    # Check readability
    if check_readable and must_exist:
        if not os.access(file_path, os.R_OK):
            raise PathValidationError(
                f"File is not readable: {file_path}\n"
                f"Please check file permissions."
            )
    
    return file_path


def validate_directory_path(
    dir_path: Union[str, Path],
    must_exist: bool = True,
    create_if_missing: bool = False
) -> Path:
    """
    Validate a directory path with security checks.
    
    Args:
        dir_path: Directory path to validate
        must_exist: Whether the directory must exist
        create_if_missing: Whether to create the directory if it doesn't exist
        
    Returns:
        Validated Path object (absolute path)
        
    Raises:
        PathValidationError: If validation fails
    """
    # Convert to Path object and resolve to absolute path
    try:
        dir_path = Path(dir_path).resolve()
    except (ValueError, OSError) as e:
        raise PathValidationError(
            f"Invalid directory path format: {dir_path}\n"
            f"Error: {e}"
        )
    
    # Check path traversal - ensure path is within allowed directories
    path_is_safe = False
    for allowed_base in ALLOWED_BASE_DIRECTORIES:
        try:
            dir_path.relative_to(allowed_base)
            path_is_safe = True
            break
        except ValueError:
            continue
    
    if not path_is_safe:
        raise PathValidationError(
            f"Path traversal detected: {dir_path}\n"
            f"Directory path must be within allowed directories:\n"
            f"{chr(10).join(str(d) for d in ALLOWED_BASE_DIRECTORIES)}"
        )
    
    # Handle directory creation
    if not dir_path.exists():
        if create_if_missing:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise PathValidationError(
                    f"Failed to create directory: {dir_path}\n"
                    f"Error: {e}"
                )
        elif must_exist:
            raise PathValidationError(
                f"Directory not found: {dir_path}\n"
                f"Please ensure the directory exists."
            )
    
    # Check if it's actually a directory
    if dir_path.exists() and not dir_path.is_dir():
        raise PathValidationError(
            f"Path is not a directory: {dir_path}\n"
            f"Expected a directory, but found a file."
        )
    
    return dir_path


def get_validated_data_file_path() -> Path:
    """
    Get the validated data file path with all security checks applied.
    
    Returns:
        Validated Path object for the data file
        
    Raises:
        PathValidationError: If validation fails
    """
    return validate_file_path(
        DATA_FILE_PATH,
        allowed_extensions=['.csv'],
        must_exist=True,
        check_readable=True
    )
