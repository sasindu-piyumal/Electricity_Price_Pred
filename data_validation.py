#!/usr/bin/env python
# coding: utf-8

"""
Data Validation Utilities
Electricity Price Prediction

This module provides robust input validation for CSV data loading to:
- Prevent injection attacks
- Handle malformed data gracefully
- Ensure data integrity
- Validate schema and data types
- Check file safety (size limits)
- Log validation failures

Security Features:
- File size limits to prevent memory exhaustion attacks
- Schema validation (required columns)
- Data type validation for numerical fields
- Range validation for prices and demands
- Graceful error handling with informative messages
- Security event logging
"""

import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE_MB = 100
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Expected schema for electricity.csv
REQUIRED_COLUMNS = [
    'DateTime', 'Holiday', 'HolidayFlag', 'DayOfWeek', 'WeekOfYear',
    'Day', 'Month', 'Year', 'PeriodOfDay', 'ForecastWindProduction',
    'SystemLoadEA', 'SMPEA', 'ORKTemperature', 'ORKWindspeed',
    'CO2Intensity', 'ActualWindProduction', 'SystemLoadEP2', 'SMPEP2'
]

# Numeric columns that must be validated
NUMERIC_COLUMNS = [
    'ForecastWindProduction', 'SystemLoadEA', 'SMPEA',
    'ORKTemperature', 'ORKWindspeed', 'CO2Intensity',
    'ActualWindProduction', 'SystemLoadEP2', 'SMPEP2'
]

# Price columns (must be >= 0)
PRICE_COLUMNS = ['SMPEA', 'SMPEP2']

# Load/demand columns (must be >= 0)
LOAD_COLUMNS = ['SystemLoadEA', 'SystemLoadEP2', 'ForecastWindProduction', 'ActualWindProduction']

# Reasonable upper bounds for validation
UPPER_BOUNDS = {
    'SMPEA': 10000,  # Price upper bound (euro/MWh)
    'SMPEP2': 10000,  # Price upper bound (euro/MWh)
    'SystemLoadEA': 100000,  # Load upper bound (MW)
    'SystemLoadEP2': 100000,  # Load upper bound (MW)
    'ForecastWindProduction': 50000,  # Wind production upper bound (MW)
    'ActualWindProduction': 50000,  # Wind production upper bound (MW)
    'CO2Intensity': 2000,  # CO2 intensity upper bound (g/kWh)
    'ORKTemperature': 50,  # Temperature upper bound (Celsius)
    'ORKWindspeed': 200,  # Wind speed upper bound (km/h)
}


class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


class SchemaValidationError(DataValidationError):
    """Exception raised when CSV schema doesn't match expected format."""
    pass


class DataTypeValidationError(DataValidationError):
    """Exception raised when data types are invalid."""
    pass


class RangeValidationError(DataValidationError):
    """Exception raised when data values are out of expected range."""
    pass


class FileSizeError(DataValidationError):
    """Exception raised when file size exceeds limit."""
    pass


def check_file_size(file_path: str, max_size_bytes: int = MAX_FILE_SIZE_BYTES) -> None:
    """
    Check if file size is within acceptable limits to prevent memory exhaustion attacks.
    
    Args:
        file_path (str): Path to the file to check
        max_size_bytes (int): Maximum allowed file size in bytes
        
    Raises:
        FileSizeError: If file size exceeds the limit
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)
    
    if not path.exists():
        error_msg = f"File not found: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    file_size = path.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    
    if file_size > max_size_bytes:
        error_msg = (
            f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed size "
            f"({max_size_bytes / (1024 * 1024):.2f} MB). "
            f"This could be a memory exhaustion attack or corrupted file."
        )
        logger.error(f"SECURITY: {error_msg}")
        raise FileSizeError(error_msg)
    
    logger.info(f"File size check passed: {file_size_mb:.2f} MB")


def validate_schema(df: pd.DataFrame, required_columns: List[str] = REQUIRED_COLUMNS) -> None:
    """
    Validate that CSV contains all required columns.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (List[str]): List of required column names
        
    Raises:
        SchemaValidationError: If required columns are missing
    """
    # Get actual columns (handling index if it was set during read_csv)
    actual_columns = df.columns.tolist()
    
    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in actual_columns and col != 'DateTime']
    
    if missing_columns:
        error_msg = (
            f"Schema validation failed: Missing required columns: {missing_columns}\n"
            f"Expected columns: {required_columns}\n"
            f"Actual columns: {actual_columns}"
        )
        logger.error(f"VALIDATION ERROR: {error_msg}")
        raise SchemaValidationError(error_msg)
    
    logger.info(f"Schema validation passed: All {len(required_columns)} required columns present")


def validate_data_types(df: pd.DataFrame, numeric_columns: List[str] = NUMERIC_COLUMNS) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validate that numeric columns contain valid numeric data.
    Attempts to convert to numeric and reports any issues.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        numeric_columns (List[str]): List of columns that should be numeric
        
    Returns:
        Tuple[pd.DataFrame, List[str]]: Validated DataFrame and list of warnings
        
    Raises:
        DataTypeValidationError: If critical data type issues are found
    """
    warnings = []
    
    for col in numeric_columns:
        if col not in df.columns:
            continue
            
        # Check if already numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        
        # Try to convert to numeric
        original_count = len(df)
        numeric_values = pd.to_numeric(df[col], errors='coerce')
        null_count = numeric_values.isna().sum()
        non_null_percentage = ((original_count - null_count) / original_count) * 100
        
        if null_count > 0:
            warning_msg = (
                f"Column '{col}': {null_count} non-numeric values found "
                f"({non_null_percentage:.1f}% valid numeric values)"
            )
            warnings.append(warning_msg)
            logger.warning(warning_msg)
        
        # If more than 50% of values are invalid, raise error
        if non_null_percentage < 50:
            error_msg = (
                f"Data type validation failed for column '{col}': "
                f"Only {non_null_percentage:.1f}% of values are valid numbers. "
                f"This indicates corrupted or malformed data."
            )
            logger.error(f"VALIDATION ERROR: {error_msg}")
            raise DataTypeValidationError(error_msg)
    
    if not warnings:
        logger.info(f"Data type validation passed: All numeric columns are valid")
    
    return df, warnings


def validate_ranges(df: pd.DataFrame) -> List[str]:
    """
    Validate that numeric values are within expected ranges.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        
    Returns:
        List[str]: List of validation warnings
        
    Raises:
        RangeValidationError: If critical range violations are found
    """
    warnings = []
    
    # Convert numeric columns to numeric (with coercion)
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Check price columns (must be >= 0)
    for col in PRICE_COLUMNS:
        if col not in df.columns:
            continue
            
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            warning_msg = f"Column '{col}': Found {negative_count} negative price values"
            warnings.append(warning_msg)
            logger.warning(warning_msg)
            
            # If more than 10% are negative, this is suspicious
            if (negative_count / len(df)) > 0.1:
                error_msg = (
                    f"Range validation failed for '{col}': "
                    f"{negative_count} negative values ({(negative_count/len(df)*100):.1f}% of data). "
                    f"Prices cannot be negative. Data may be corrupted."
                )
                logger.error(f"VALIDATION ERROR: {error_msg}")
                raise RangeValidationError(error_msg)
    
    # Check load/demand columns (must be >= 0)
    for col in LOAD_COLUMNS:
        if col not in df.columns:
            continue
            
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            warning_msg = f"Column '{col}': Found {negative_count} negative load/demand values"
            warnings.append(warning_msg)
            logger.warning(warning_msg)
            
            # If more than 10% are negative, this is suspicious
            if (negative_count / len(df)) > 0.1:
                error_msg = (
                    f"Range validation failed for '{col}': "
                    f"{negative_count} negative values ({(negative_count/len(df)*100):.1f}% of data). "
                    f"Load/demand cannot be negative. Data may be corrupted."
                )
                logger.error(f"VALIDATION ERROR: {error_msg}")
                raise RangeValidationError(error_msg)
    
    # Check upper bounds
    for col, upper_bound in UPPER_BOUNDS.items():
        if col not in df.columns:
            continue
            
        exceeding_count = (df[col] > upper_bound).sum()
        if exceeding_count > 0:
            warning_msg = (
                f"Column '{col}': {exceeding_count} values exceed upper bound "
                f"of {upper_bound}"
            )
            warnings.append(warning_msg)
            logger.warning(warning_msg)
            
            # If more than 20% exceed bounds, data might be corrupted
            if (exceeding_count / len(df)) > 0.2:
                error_msg = (
                    f"Range validation failed for '{col}': "
                    f"{exceeding_count} values ({(exceeding_count/len(df)*100):.1f}% of data) "
                    f"exceed reasonable upper bound of {upper_bound}. "
                    f"Data may be corrupted or in unexpected units."
                )
                logger.error(f"VALIDATION ERROR: {error_msg}")
                raise RangeValidationError(error_msg)
    
    if not warnings:
        logger.info("Range validation passed: All values within expected ranges")
    
    return warnings


def validate_csv_file(file_path: str, 
                      validate_file_size: bool = True,
                      validate_columns: bool = True,
                      validate_types: bool = True,
                      validate_value_ranges: bool = True) -> pd.DataFrame:
    """
    Main validation function: Performs comprehensive validation on CSV file.
    
    This function:
    1. Checks file size to prevent memory exhaustion
    2. Validates CSV can be parsed
    3. Validates schema (required columns)
    4. Validates data types for numeric columns
    5. Validates value ranges
    
    Args:
        file_path (str): Path to CSV file
        validate_file_size (bool): Whether to check file size
        validate_columns (bool): Whether to validate schema
        validate_types (bool): Whether to validate data types
        validate_value_ranges (bool): Whether to validate value ranges
        
    Returns:
        pd.DataFrame: Validated DataFrame
        
    Raises:
        DataValidationError: If validation fails
        FileNotFoundError: If file doesn't exist
    """
    logger.info(f"Starting validation for file: {file_path}")
    
    try:
        # Step 1: Check file size
        if validate_file_size:
            logger.info("Step 1/5: Checking file size...")
            check_file_size(file_path)
        
        # Step 2: Try to read CSV with error handling
        logger.info("Step 2/5: Reading CSV file...")
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=[0])
        except pd.errors.ParserError as e:
            error_msg = f"CSV parsing failed: {str(e)}. File may be corrupted or not a valid CSV."
            logger.error(f"VALIDATION ERROR: {error_msg}")
            raise DataValidationError(error_msg)
        except Exception as e:
            error_msg = f"Failed to read CSV file: {str(e)}"
            logger.error(f"VALIDATION ERROR: {error_msg}")
            raise DataValidationError(error_msg)
        
        logger.info(f"CSV loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Step 3: Validate schema
        if validate_columns:
            logger.info("Step 3/5: Validating schema...")
            validate_schema(df)
        
        # Step 4: Validate data types
        if validate_types:
            logger.info("Step 4/5: Validating data types...")
            df, type_warnings = validate_data_types(df)
        
        # Step 5: Validate ranges
        if validate_value_ranges:
            logger.info("Step 5/5: Validating value ranges...")
            range_warnings = validate_ranges(df)
        
        logger.info("="*60)
        logger.info("VALIDATION SUCCESSFUL")
        logger.info(f"File: {file_path}")
        logger.info(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        logger.info("="*60)
        
        return df
        
    except DataValidationError:
        # Re-raise validation errors
        raise
    except Exception as e:
        error_msg = f"Unexpected error during validation: {str(e)}"
        logger.error(f"VALIDATION ERROR: {error_msg}")
        raise DataValidationError(error_msg)


def safe_read_csv(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Safely read and validate CSV file with all security checks.
    
    This is a drop-in replacement for pd.read_csv() with built-in validation.
    
    Args:
        file_path (str): Path to CSV file
        **kwargs: Additional arguments passed to pd.read_csv (except index_col and parse_dates)
        
    Returns:
        pd.DataFrame: Validated DataFrame
        
    Raises:
        DataValidationError: If validation fails
        
    Example:
        >>> df = safe_read_csv('electricity.csv')
    """
    # Validate the file first
    df = validate_csv_file(file_path)
    return df


# Example usage and testing
if __name__ == "__main__":
    print("Data Validation Module - Security Tests")
    print("=" * 60)
    
    # Test with actual electricity.csv if it exists
    test_file = "./electricity.csv"
    
    if Path(test_file).exists():
        print(f"\nTesting validation with: {test_file}")
        try:
            df = validate_csv_file(test_file)
            print(f"\n✓ Validation successful!")
            print(f"  Rows: {df.shape[0]}")
            print(f"  Columns: {df.shape[1]}")
            print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        except DataValidationError as e:
            print(f"\n✗ Validation failed: {e}")
    else:
        print(f"\n⊘ Test file not found: {test_file}")
    
    print("\n" + "=" * 60)
