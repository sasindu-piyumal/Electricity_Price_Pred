# Configuration Guide

## Overview

The `config.py` module provides centralized configuration management for the Electricity Price Prediction project. It supports loading configuration from `.env` files and environment variables, with built-in validation.

## Quick Start

### Basic Usage

```python
from config import DATA_FILE_PATH, ENVIRONMENT, TEST_SIZE

# Use the configuration values
data = pd.read_csv(DATA_FILE_PATH)
```

### Setting Up Configuration

1. **Copy the example configuration:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` file with your values:**
   ```env
   ENVIRONMENT=development
   DATA_FILE_PATH=./electricity.csv
   TEST_SIZE=0.2
   RANDOM_STATE=42
   ```

3. **Or use environment variables:**
   ```bash
   export DATA_FILE_PATH="/path/to/your/data.csv"
   export ENVIRONMENT=production
   ```

## Configuration Options

### Environment Settings

- **ENVIRONMENT** (default: `development`)
  - Allowed values: `development`, `production`
  - Controls environment-specific behavior

### File Paths

- **DATA_FILE_PATH** (default: `./electricity.csv`)
  - Path to the electricity dataset CSV file
  - Must exist and be readable
  - Can be absolute or relative path

### Model Parameters

- **TEST_SIZE** (default: `0.2`)
  - Train/test split ratio
  - Range: 0.0 to 1.0

- **RANDOM_STATE** (default: `42`)
  - Random seed for reproducibility
  - Must be >= 0

- **N_ESTIMATORS** (default: `100`)
  - Number of estimators for Random Forest
  - Range: 1 to 10000

### Logging

- **LOG_LEVEL** (default: `INFO`)
  - Allowed values: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

- **DEBUG_MODE** (default: `false`)
  - Enable/disable debug mode
  - Accepts: true/false, yes/no, 1/0, on/off

## Available Configuration Values

```python
from config import (
    # Environment
    ENVIRONMENT,        # Current environment (development/production)
    IS_DEVELOPMENT,     # Boolean: True if in development
    IS_PRODUCTION,      # Boolean: True if in production
    
    # File Paths
    DATA_FILE_PATH,     # Path object to data file
    
    # Model Parameters
    TEST_SIZE,          # Train/test split ratio
    RANDOM_STATE,       # Random seed
    N_ESTIMATORS,       # Number of estimators
    
    # Logging
    LOG_LEVEL,          # Logging level
    DEBUG_MODE,         # Debug mode flag
)
```

## Validation Features

The configuration module automatically validates:

1. **Path Validation**
   - Checks if paths exist
   - Verifies read permissions
   - Provides clear error messages if files are missing

2. **Numeric Validation**
   - Ensures values are numbers
   - Validates ranges (min/max)
   - Converts to appropriate types (int/float)

3. **Enum Validation**
   - Checks against allowed values
   - Case-insensitive matching
   - Clear error messages with valid options

4. **Boolean Validation**
   - Accepts multiple formats (true/false, yes/no, 1/0, on/off)
   - Converts to Python boolean

## Error Handling

The module provides clear error messages:

```python
# Example error messages:
ConfigurationError: Cannot find data file. Tried:
  1. Configured path: /path/to/missing/file.csv
  2. Default path: ./electricity.csv
Please ensure electricity.csv exists or set DATA_FILE_PATH correctly.

ValidationError: Invalid value 'staging'. Allowed values: development, production

ValidationError: Value 1.5 exceeds maximum allowed value 1.0
```

## Environment Variable Precedence

Configuration is loaded in this order (later overrides earlier):

1. Default values in `config.py`
2. Values from `.env` file
3. Environment variables (highest priority)

## Development vs Production

The module automatically detects the environment:

```python
from config import IS_DEVELOPMENT, IS_PRODUCTION

if IS_DEVELOPMENT:
    # Enable debug features
    print_config_summary()

if IS_PRODUCTION:
    # Use production settings
    pass
```

## Debugging Configuration

Print current configuration:

```python
from config import print_config_summary

print_config_summary()
```

Output:
```
======================================================================
Configuration Summary
======================================================================
Environment:        development
Data File Path:     /path/to/electricity.csv
Test Size:          0.2
Random State:       42
N Estimators:       100
Log Level:          INFO
Debug Mode:         False
======================================================================
```

## Troubleshooting

### "Cannot find data file" Error

1. Check if `electricity.csv` exists in project root
2. Set `DATA_FILE_PATH` in `.env` file or environment variable
3. Ensure file has read permissions

### Invalid Configuration Values

1. Check `.env` file for typos
2. Verify values are in allowed ranges
3. Check error messages for valid options

### Configuration Not Loading

1. Ensure `.env` file is in project root
2. Check file permissions
3. Verify environment variables are exported
4. Check for syntax errors in `.env` file

## Best Practices

1. **Use `.env` for local development**
   - Keep `.env` in `.gitignore`
   - Share `.env.example` with team

2. **Use environment variables for production**
   - Set via deployment platform
   - Keep sensitive values secure

3. **Validate early**
   - Import config at application start
   - Fail fast on configuration errors

4. **Document custom values**
   - Update `.env.example` when adding new config
   - Document valid ranges and options

## Example: Migrating Hardcoded Values

Before:
```python
data = pd.read_csv("Z:\\Sasindu\\Data set\\electricity.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

After:
```python
from config import DATA_FILE_PATH, TEST_SIZE, RANDOM_STATE

data = pd.read_csv(DATA_FILE_PATH)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
```

## Support

For issues or questions:
1. Check error messages for guidance
2. Verify `.env.example` for correct format
3. Review this guide for common solutions
