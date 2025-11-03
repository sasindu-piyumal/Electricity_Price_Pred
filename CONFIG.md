# Configuration Guide

This document provides detailed information about all configuration parameters for the Electricity Price Prediction project.

## Quick Start

1. Copy the example configuration file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your specific values:
   ```bash
   nano .env  # or use your preferred text editor
   ```

3. Ensure `.env` is never committed to version control (already included in `.gitignore`)

## Configuration Format

The project uses `.env` file format for configuration management. This format is:
- Simple key-value pairs
- Easy to read and maintain
- Compatible with environment variables
- Suitable for both development and production environments

## Configuration Parameters

### Data Configuration

#### `DATA_FILE_PATH`
- **Type**: String (file path)
- **Required**: Yes
- **Default**: `./electricity.csv`
- **Description**: Path to the electricity dataset CSV file containing the training data.
- **Examples**:
  - Relative path: `./data/electricity.csv`
  - Absolute path: `/home/user/projects/data/electricity.csv`
  - Windows path: `C:\Users\username\data\electricity.csv`
- **Notes**: 
  - The file must exist and be readable
  - Should point to a valid CSV file with the expected schema
  - Currently expects columns like: DateTime, HolidayFlag, DayOfWeek, WeekOfYear, etc.

### Output Configuration

#### `OUTPUT_DIR`
- **Type**: String (directory path)
- **Required**: Yes
- **Default**: `./output`
- **Description**: Directory where plots, results, and model outputs will be saved.
- **Examples**:
  - Relative path: `./results`
  - Absolute path: `/home/user/projects/output`
- **Notes**:
  - Directory will be created automatically if it doesn't exist
  - Ensure write permissions for the specified directory
  - Contents may include: plots (PNG/PDF), model files, prediction results

### Logging Configuration

#### `LOG_LEVEL`
- **Type**: String (enumerated)
- **Required**: Yes
- **Default**: `INFO`
- **Valid Values**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- **Description**: Controls the verbosity of application logging.
- **Value Descriptions**:
  - `DEBUG`: Most verbose. Shows detailed diagnostic information useful for troubleshooting.
  - `INFO`: General informational messages about program execution and progress.
  - `WARNING`: Alerts about potential issues that don't stop execution.
  - `ERROR`: Serious problems that prevented specific operations from completing.
  - `CRITICAL`: Very serious errors that may cause the program to terminate.
- **Recommendations**:
  - Development: Use `DEBUG` or `INFO`
  - Production: Use `WARNING` or `ERROR`
- **Notes**:
  - Lower levels include all higher level messages (e.g., ERROR level includes CRITICAL)
  - Debug level may significantly increase log file sizes

### Validation Configuration

#### `MAX_FILE_SIZE_MB`
- **Type**: Integer (positive number)
- **Required**: Yes
- **Default**: `500`
- **Description**: Maximum allowed CSV file size in megabytes for validation before processing.
- **Examples**:
  - `100` - 100 MB limit
  - `1000` - 1 GB limit
  - `0` - Disable file size validation
- **Notes**:
  - Prevents accidental processing of unexpectedly large files
  - Set based on your system's available memory
  - Set to `0` to disable this validation (not recommended)
  - Consider your system's RAM when setting this value

### Environment Configuration

#### `ENVIRONMENT`
- **Type**: String (enumerated)
- **Required**: Yes
- **Default**: `DEV`
- **Valid Values**: `DEV`, `PROD`
- **Description**: Indicates the environment in which the application is running.
- **Value Descriptions**:
  - `DEV`: Development environment
    - May enable additional debug features
    - More verbose logging by default
    - Relaxed validation rules
    - Development-friendly error messages
  - `PROD`: Production environment
    - Optimized for performance
    - Reduced logging overhead
    - Strict validation
    - User-friendly error messages
- **Notes**:
  - Affects application behavior and defaults
  - Always use `PROD` for production deployments

## Example Configurations

### Development Environment

```bash
DATA_FILE_PATH=./electricity.csv
OUTPUT_DIR=./output
LOG_LEVEL=DEBUG
MAX_FILE_SIZE_MB=500
ENVIRONMENT=DEV
```

### Production Environment

```bash
DATA_FILE_PATH=/data/production/electricity.csv
OUTPUT_DIR=/var/app/output
LOG_LEVEL=WARNING
MAX_FILE_SIZE_MB=1000
ENVIRONMENT=PROD
```

### Testing Environment

```bash
DATA_FILE_PATH=./test_data/electricity_sample.csv
OUTPUT_DIR=./test_output
LOG_LEVEL=INFO
MAX_FILE_SIZE_MB=100
ENVIRONMENT=DEV
```

## Best Practices

1. **Never Commit `.env` Files**: Always keep actual configuration files out of version control
2. **Use `.env.example` as Template**: Keep it updated when adding new parameters
3. **Environment-Specific Files**: Consider using `.env.dev`, `.env.prod` for different environments
4. **Secure Sensitive Data**: If adding API keys or passwords, ensure proper file permissions
5. **Document Changes**: Update this CONFIG.md when adding or modifying parameters
6. **Validate on Startup**: Application should validate all required parameters on initialization
7. **Default Values**: Provide sensible defaults for non-critical parameters

## Troubleshooting

### Configuration File Not Found
- Ensure `.env` file exists in the project root directory
- Check that you've copied from `.env.example`
- Verify file permissions (should be readable)

### Invalid Parameter Values
- Review valid values for enumerated parameters (LOG_LEVEL, ENVIRONMENT)
- Check file paths for typos or incorrect formatting
- Ensure numeric values are positive integers where required

### File Path Issues
- Use absolute paths if relative paths cause issues
- On Windows, use forward slashes or escaped backslashes in paths
- Verify that specified files/directories exist and are accessible

### Permission Errors
- Ensure read permissions for DATA_FILE_PATH
- Ensure write permissions for OUTPUT_DIR
- Check directory ownership and access rights

## Future Configuration Parameters

The following parameters may be added in future versions:
- Model hyperparameters (n_estimators, max_depth, etc.)
- Database connection strings
- API endpoints for external services
- Caching configuration
- Parallel processing settings

## Support

For questions or issues related to configuration, please refer to the main project documentation or contact the development team.
