"""
Configuration Loader Module
============================

This module provides centralized configuration management for the electricity price
prediction project. It loads configuration from .env files and environment variables,
with environment variables taking precedence.

Usage:
    from config import DATA_FILE_PATH, ENVIRONMENT, TEST_SIZE

Features:
    - Loads from .env file (if exists) and environment variables
    - Environment variables override .env file values
    - Provides sensible default values for all parameters
    - Validates configuration at import time
    - Supports development and production environments
    - Clear error messages for configuration issues

Configuration Values:
    - DATA_FILE_PATH: Path to the electricity dataset CSV file
    - ENVIRONMENT: Application environment (development or production)
    - TEST_SIZE: Train/test split ratio for model validation
    - RANDOM_STATE: Random seed for reproducibility
    - N_ESTIMATORS: Number of estimators for Random Forest model
    - LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - DEBUG_MODE: Enable debug mode flag

Environment Variables:
    All configuration can be overridden via environment variables with the same names.
"""

import os
import sys
from pathlib import Path
from typing import Union, Any, List
from dotenv import load_dotenv


# ============================================================================
# Custom Exceptions
# ============================================================================

class ConfigurationError(Exception):
    """Base exception for configuration-related errors."""
    pass


class ValidationError(ConfigurationError):
    """Raised when configuration validation fails."""
    pass


class MissingConfigError(ConfigurationError):
    """Raised when required configuration is missing."""
    pass


# ============================================================================
# Validation Functions
# ============================================================================

def validate_path(path: str, must_exist: bool = True, must_be_readable: bool = True) -> Path:
    """
    Validate a file or directory path.
    
    Args:
        path: Path string to validate
        must_exist: If True, path must exist
        must_be_readable: If True, path must be readable
        
    Returns:
        Path object if valid
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        path_obj = Path(path).resolve()
    except Exception as e:
        raise ValidationError(f"Invalid path '{path}': {str(e)}")
    
    if must_exist and not path_obj.exists():
        raise ValidationError(
            f"Path does not exist: {path_obj}\n"
            f"Please ensure the file exists or update the DATA_FILE_PATH configuration."
        )
    
    if must_be_readable and path_obj.exists():
        if not os.access(path_obj, os.R_OK):
            raise ValidationError(f"Path is not readable: {path_obj}")
    
    return path_obj


def validate_number(value: Any, min_val: float = None, max_val: float = None, 
                   allow_none: bool = False) -> Union[float, int, None]:
    """
    Validate a numeric parameter.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        allow_none: If True, None is accepted
        
    Returns:
        Validated numeric value
        
    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        if allow_none:
            return None
        raise ValidationError("Value cannot be None")
    
    try:
        # Try to convert to float first
        num_val = float(value)
        
        # Convert to int if it's a whole number
        if num_val.is_integer():
            num_val = int(num_val)
    except (ValueError, TypeError) as e:
        raise ValidationError(f"Invalid numeric value '{value}': {str(e)}")
    
    if min_val is not None and num_val < min_val:
        raise ValidationError(
            f"Value {num_val} is below minimum allowed value {min_val}"
        )
    
    if max_val is not None and num_val > max_val:
        raise ValidationError(
            f"Value {num_val} exceeds maximum allowed value {max_val}"
        )
    
    return num_val


def validate_enum(value: Any, allowed_values: List[str], case_sensitive: bool = False) -> str:
    """
    Validate an enum-style value against allowed options.
    
    Args:
        value: Value to validate
        allowed_values: List of allowed values
        case_sensitive: If True, comparison is case-sensitive
        
    Returns:
        Validated value (in original case from allowed_values)
        
    Raises:
        ValidationError: If value not in allowed values
    """
    if not isinstance(value, str):
        raise ValidationError(f"Expected string value, got {type(value).__name__}")
    
    if case_sensitive:
        if value not in allowed_values:
            raise ValidationError(
                f"Invalid value '{value}'. Allowed values: {', '.join(allowed_values)}"
            )
        return value
    else:
        # Case-insensitive comparison
        value_lower = value.lower()
        for allowed in allowed_values:
            if allowed.lower() == value_lower:
                return allowed
        
        raise ValidationError(
            f"Invalid value '{value}'. Allowed values: {', '.join(allowed_values)}"
        )


def validate_boolean(value: Any) -> bool:
    """
    Validate and convert a boolean value.
    
    Accepts: True/False, true/false, yes/no, 1/0, on/off
    
    Args:
        value: Value to validate and convert
        
    Returns:
        Boolean value
        
    Raises:
        ValidationError: If value cannot be converted to boolean
    """
    if isinstance(value, bool):
        return value
    
    if isinstance(value, (int, float)):
        return bool(value)
    
    if isinstance(value, str):
        value_lower = value.lower().strip()
        if value_lower in ('true', 'yes', '1', 'on'):
            return True
        elif value_lower in ('false', 'no', '0', 'off', ''):
            return False
    
    raise ValidationError(
        f"Cannot convert '{value}' to boolean. "
        f"Use: true/false, yes/no, 1/0, on/off"
    )


# ============================================================================
# Configuration Loading
# ============================================================================

def get_config_value(key: str, default: Any = None, required: bool = False) -> Any:
    """
    Get configuration value from environment variables or default.
    
    Environment variables take precedence over defaults.
    
    Args:
        key: Configuration key name
        default: Default value if not found
        required: If True, raises error if value not found
        
    Returns:
        Configuration value
        
    Raises:
        MissingConfigError: If required value is missing
    """
    value = os.getenv(key, default)
    
    if value is None and required:
        raise MissingConfigError(
            f"Required configuration '{key}' is missing. "
            f"Please set it in .env file or as an environment variable."
        )
    
    return value


# ============================================================================
# Load Environment Configuration
# ============================================================================

# Load .env file if it exists (doesn't override existing env vars)
_env_file_path = Path('.env')
if _env_file_path.exists():
    load_dotenv(_env_file_path)


# ============================================================================
# Environment Detection
# ============================================================================

try:
    ENVIRONMENT = validate_enum(
        get_config_value('ENVIRONMENT', 'development'),
        allowed_values=['development', 'production'],
        case_sensitive=False
    )
except ValidationError as e:
    print(f"Configuration Error: {e}", file=sys.stderr)
    print("Defaulting to 'development' environment.", file=sys.stderr)
    ENVIRONMENT = 'development'

IS_DEVELOPMENT = ENVIRONMENT.lower() == 'development'
IS_PRODUCTION = ENVIRONMENT.lower() == 'production'


# ============================================================================
# File Paths Configuration
# ============================================================================

# Default data file path (relative to project root)
_default_data_path = './electricity.csv'

try:
    _data_path_str = get_config_value('DATA_FILE_PATH', _default_data_path)
    DATA_FILE_PATH = validate_path(
        _data_path_str,
        must_exist=True,
        must_be_readable=True
    )
except ValidationError as e:
    print(f"Configuration Error: {e}", file=sys.stderr)
    print(f"Attempting to use default path: {_default_data_path}", file=sys.stderr)
    
    try:
        # Try default path
        DATA_FILE_PATH = validate_path(
            _default_data_path,
            must_exist=True,
            must_be_readable=True
        )
    except ValidationError:
        # If default also fails, provide clear error
        raise ConfigurationError(
            f"Cannot find data file. Tried:\n"
            f"  1. Configured path: {_data_path_str}\n"
            f"  2. Default path: {_default_data_path}\n"
            f"Please ensure electricity.csv exists or set DATA_FILE_PATH correctly."
        )


# ============================================================================
# Model Parameters Configuration
# ============================================================================

try:
    TEST_SIZE = validate_number(
        get_config_value('TEST_SIZE', 0.2),
        min_val=0.0,
        max_val=1.0
    )
except ValidationError as e:
    print(f"Configuration Warning: Invalid TEST_SIZE - {e}", file=sys.stderr)
    print("Using default value: 0.2", file=sys.stderr)
    TEST_SIZE = 0.2

try:
    RANDOM_STATE = validate_number(
        get_config_value('RANDOM_STATE', 42),
        min_val=0
    )
    RANDOM_STATE = int(RANDOM_STATE)
except ValidationError as e:
    print(f"Configuration Warning: Invalid RANDOM_STATE - {e}", file=sys.stderr)
    print("Using default value: 42", file=sys.stderr)
    RANDOM_STATE = 42

try:
    N_ESTIMATORS = validate_number(
        get_config_value('N_ESTIMATORS', 100),
        min_val=1,
        max_val=10000
    )
    N_ESTIMATORS = int(N_ESTIMATORS)
except ValidationError as e:
    print(f"Configuration Warning: Invalid N_ESTIMATORS - {e}", file=sys.stderr)
    print("Using default value: 100", file=sys.stderr)
    N_ESTIMATORS = 100


# ============================================================================
# Logging Configuration
# ============================================================================

try:
    LOG_LEVEL = validate_enum(
        get_config_value('LOG_LEVEL', 'INFO'),
        allowed_values=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        case_sensitive=False
    ).upper()
except ValidationError as e:
    print(f"Configuration Warning: Invalid LOG_LEVEL - {e}", file=sys.stderr)
    print("Using default value: INFO", file=sys.stderr)
    LOG_LEVEL = 'INFO'

try:
    DEBUG_MODE = validate_boolean(
        get_config_value('DEBUG_MODE', 'false')
    )
except ValidationError as e:
    print(f"Configuration Warning: Invalid DEBUG_MODE - {e}", file=sys.stderr)
    print("Using default value: False", file=sys.stderr)
    DEBUG_MODE = False


# ============================================================================
# Configuration Summary (for debugging)
# ============================================================================

def print_config_summary():
    """Print a summary of the current configuration."""
    print("=" * 70)
    print("Configuration Summary")
    print("=" * 70)
    print(f"Environment:        {ENVIRONMENT}")
    print(f"Data File Path:     {DATA_FILE_PATH}")
    print(f"Test Size:          {TEST_SIZE}")
    print(f"Random State:       {RANDOM_STATE}")
    print(f"N Estimators:       {N_ESTIMATORS}")
    print(f"Log Level:          {LOG_LEVEL}")
    print(f"Debug Mode:         {DEBUG_MODE}")
    print("=" * 70)


# Print configuration summary in development mode with debug enabled
if IS_DEVELOPMENT and DEBUG_MODE:
    print_config_summary()


# ============================================================================
# Export Public API
# ============================================================================

__all__ = [
    # Environment
    'ENVIRONMENT',
    'IS_DEVELOPMENT',
    'IS_PRODUCTION',
    
    # File Paths
    'DATA_FILE_PATH',
    
    # Model Parameters
    'TEST_SIZE',
    'RANDOM_STATE',
    'N_ESTIMATORS',
    
    # Logging
    'LOG_LEVEL',
    'DEBUG_MODE',
    
    # Utilities
    'print_config_summary',
    
    # Exceptions
    'ConfigurationError',
    'ValidationError',
    'MissingConfigError',
]
