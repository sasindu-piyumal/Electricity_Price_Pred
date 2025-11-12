# Electricity Price Prediction Model

A machine learning project for predicting electricity prices using Random Forest regression with comprehensive hyperparameter tuning.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependency Management](#dependency-management)
- [Security](#security)
- [Contributing](#contributing)

## Overview

This project implements a Random Forest regression model to predict electricity prices (SMPEP2) based on various features including:
- System load and forecasts
- Wind production (actual and forecasted)
- Temperature and wind speed measurements
- CO2 intensity
- Temporal features (day, month, period of day)

The baseline model achieves an R² score of 0.6502, with hyperparameter tuning implemented to improve performance.

## Requirements

### Python Version

- **Python 3.8.18 or higher** (tested on Python 3.8.18)
- See `runtime.txt` for the recommended Python version

### System Requirements

- Minimum 4GB RAM (8GB recommended for hyperparameter tuning)
- ~500MB disk space for dependencies

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <project-directory>
```

### 2. Create a Virtual Environment

Using `venv` (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Using `conda`:

```bash
conda create -n electricity-prediction python=3.8
conda activate electricity-prediction
```

Using `pyenv` (if you have `.python-version` file):

```bash
pyenv install 3.8.18
pyenv local 3.8.18
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

**For production use:**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**For development (includes testing, linting, and Jupyter):**

```bash
pip install --upgrade pip
pip install -r requirements-dev.txt
```

### 4. Configure Data Path (Optional)

The project uses environment variables for secure, configurable file path handling:

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env to set your data file path (optional, defaults to ./electricity.csv)
# DATA_PATH=./electricity.csv
```

**Default behavior:** If no `.env` file exists or `DATA_PATH` is not set, the scripts will automatically use `./electricity.csv` in the project root.

### 5. Verify Installation

```bash
python -c "import pandas, numpy, sklearn, matplotlib, seaborn, statsmodels; print('All dependencies installed successfully!')"
```

## Usage

### Configuring Data File Path

The project uses secure, configurable path handling to improve portability and security:

**Method 1: Environment Variable (Recommended)**

```bash
# Set DATA_PATH environment variable
export DATA_PATH=./electricity.csv  # Linux/macOS
set DATA_PATH=./electricity.csv     # Windows CMD
$env:DATA_PATH="./electricity.csv"  # Windows PowerShell
```

**Method 2: .env File (Persistent Configuration)**

```bash
# Create .env file from template
cp .env.example .env

# Edit .env file to set your custom path
# DATA_PATH=./electricity.csv
# DATA_PATH=/absolute/path/to/electricity.csv
# DATA_PATH=C:\Users\YourName\data\electricity.csv
```

**Method 3: Use Default**

If neither environment variable nor `.env` file is configured, the scripts automatically use `./electricity.csv` in the project root.

### Running the Main Analysis

```bash
python "Electricity Data.py"
```

Or with custom data path:

```bash
DATA_PATH=./my_data.csv python "Electricity Data.py"
```

### Running Hyperparameter Tuning

```bash
python hyperparameter_tuning.py
```

**Note:** This process may take 1-2 hours depending on your hardware.

### Analyzing Tuning Results

```bash
python analyze_tuning_results.py
```

### Using Jupyter Notebook

```bash
jupyter notebook "Electricity Data.ipynb"
```

Or with JupyterLab:

```bash
jupyter lab
```

## Project Structure

```
.
├── electricity.csv                      # Dataset
├── Electricity Data.py                  # Main analysis script
├── Electricity Data.ipynb               # Jupyter notebook version
├── hyperparameter_tuning.py             # Hyperparameter optimization
├── analyze_tuning_results.py            # Results analysis utility
├── path_utils.py                        # Secure path handling utilities
├── requirements.txt                     # Production dependencies
├── requirements-dev.txt                 # Development dependencies
├── runtime.txt                          # Python version specification
├── .python-version                      # pyenv version file
├── .env.example                         # Environment variable template
├── .gitignore                           # Git ignore rules
├── README.md                            # This file
├── HYPERPARAMETER_TUNING_README.md      # Tuning documentation
├── IMPLEMENTATION_SUMMARY.md            # Implementation details
└── Electricity Price Prediction Model.pdf  # Project documentation
```

## Configuration

### Data File Path Configuration

The project implements secure, configurable file path handling with the following features:

**Security Features:**
- ✅ **Directory Traversal Protection**: Prevents `../` attacks
- ✅ **Path Validation**: Ensures files exist and are within allowed directories
- ✅ **Cross-Platform Support**: Works on Windows, Linux, and macOS
- ✅ **Environment Variable Support**: Configurable via `.env` file

**Configuration Options:**

1. **Default (No Configuration Required)**
   - Automatically uses `./electricity.csv` in project root
   - No setup needed if data file is in standard location

2. **Environment Variable**
   ```bash
   export DATA_PATH=/path/to/your/electricity.csv
   python "Electricity Data.py"
   ```

3. **.env File (Recommended for Development)**
   ```bash
   # Copy template
   cp .env.example .env
   
   # Edit .env file
   DATA_PATH=./electricity.csv
   ```

**Path Validation:**

The `path_utils.py` module validates all file paths to ensure:
- Paths exist and are readable
- Relative paths don't escape the project directory
- Symlinks are resolved to actual paths
- Absolute paths are allowed but validated

**Testing Path Security:**

```bash
# Test the path utilities
python path_utils.py

# This will run security tests including:
# - Default path loading
# - Directory traversal attack prevention
# - Absolute path validation
# - Relative path validation
```

## Dependency Management

### Production Dependencies

All production dependencies are pinned to specific versions in `requirements.txt`:

- **pandas 2.0.3** - Data manipulation and analysis
- **numpy 1.24.3** - Numerical computing
- **scikit-learn 1.3.0** - Machine learning algorithms
- **statsmodels 0.14.0** - Statistical modeling
- **matplotlib 3.7.2** - Data visualization
- **seaborn 0.12.2** - Statistical data visualization
- **python-dotenv 1.0.0** - Environment variable management

### Development Dependencies

Development tools are specified in `requirements-dev.txt`:

- **Jupyter/Notebook** - Interactive development environment
- **pytest** - Testing framework
- **black, flake8, pylint** - Code quality tools
- **pip-audit, safety** - Security vulnerability scanning
- **sphinx** - Documentation generation

### Updating Dependencies

#### Security Updates

To update dependencies for security patches while maintaining minor version compatibility:

```bash
# Check for vulnerabilities first
pip-audit

# Update specific package
pip install --upgrade package-name==X.Y.Z

# Update requirements file
pip freeze | grep package-name > temp.txt
# Manually update requirements.txt with new version
```

#### Major Version Upgrades

1. Create a new branch for testing:
   ```bash
   git checkout -b dependency-upgrade
   ```

2. Update one dependency at a time:
   ```bash
   pip install --upgrade package-name
   ```

3. Run all scripts to verify compatibility:
   ```bash
   python "Electricity Data.py"
   python hyperparameter_tuning.py
   python analyze_tuning_results.py
   ```

4. Update requirements files:
   ```bash
   pip freeze > requirements-new.txt
   # Review and merge changes into requirements.txt
   ```

5. Commit and test thoroughly before merging.

### Reproducible Builds

To ensure reproducible builds across different environments:

1. **Always use pinned versions** - Specified in requirements.txt
2. **Use virtual environments** - Isolate project dependencies
3. **Document Python version** - Specified in runtime.txt and .python-version
4. **Lock dependency tree** - Consider using `pip-tools` for complete dependency locking:

```bash
pip install pip-tools
pip-compile requirements.in  # Generates requirements.txt with full dependency tree
```

## Security

### Secure File Path Handling

The project implements secure file path handling to prevent common vulnerabilities:

**Protection Against Directory Traversal Attacks:**

The `path_utils.py` module prevents malicious path inputs like:
- `../../../etc/passwd` (Unix)
- `..\..\..\..\Windows\System32\config\SAM` (Windows)
- Symlink attacks
- Path injection

**How It Works:**

1. All file paths go through `validate_path()` function
2. Paths are resolved to absolute paths
3. Relative paths are checked to ensure they don't escape project directory
4. File existence is verified
5. Only files (not directories) are allowed

**Example:**

```python
from path_utils import get_safe_path_str

# Safe - uses validated path
data = pd.read_csv(get_safe_path_str())

# The following would be rejected with ValueError:
# DATA_PATH=../../../etc/passwd
```

### Vulnerability Scanning

This project includes automated security scanning tools to detect known vulnerabilities in dependencies.

#### Using pip-audit (Recommended)

```bash
# Install development dependencies first
pip install -r requirements-dev.txt

# Run security audit
pip-audit

# Generate detailed report
pip-audit --format json > security-report.json

# Fix vulnerabilities automatically (when possible)
pip-audit --fix
```

#### Using safety

```bash
# Check for known security vulnerabilities
safety check

# Check with detailed output
safety check --full-report

# Use with CI/CD
safety check --json
```

#### Automated Scanning

For continuous security monitoring, run pip-audit regularly:

```bash
# Weekly security check (add to cron or CI/CD)
pip-audit --require-hashes --format cyclonedx-json
```

### Security Best Practices

1. **Regular Updates**: Check for security updates weekly
2. **Review Changes**: Always review dependency updates before applying
3. **Audit Before Deployment**: Run `pip-audit` before deploying to production
4. **Monitor Advisories**: Subscribe to security advisories for critical packages
5. **Use Virtual Environments**: Never install packages system-wide

### Known Issues

Check the security report regularly:

```bash
pip-audit --desc
```

For any HIGH or CRITICAL vulnerabilities, prioritize immediate updates.

## Environment Setup for Different Scenarios

### Local Development

```bash
# Using pyenv (recommended for version management)
pyenv install 3.8.18
pyenv local 3.8.18
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
```

### Production Deployment

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### CI/CD Pipeline

```bash
# Example for GitHub Actions, GitLab CI, etc.
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pip-audit
pip-audit
python -m pytest  # If tests are added
```

### Docker Container

```dockerfile
FROM python:3.8.18-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "hyperparameter_tuning.py"]
```

## Troubleshooting

### Common Issues

**ImportError: No module named 'X'**
- Solution: Ensure you've activated the virtual environment and installed dependencies
  ```bash
  source venv/bin/activate  # or venv\Scripts\activate on Windows
  pip install -r requirements.txt
  ```

**Version Conflicts**
- Solution: Start with a fresh virtual environment
  ```bash
  deactivate
  rm -rf venv
  python -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```

**Jupyter Kernel Issues**
- Solution: Install and register the kernel
  ```bash
  pip install ipykernel
  python -m ipykernel install --user --name=electricity-prediction
  ```

**pip-audit fails**
- Solution: Ensure you have the latest pip version
  ```bash
  pip install --upgrade pip pip-audit
  ```

## Contributing

### Development Setup

1. Fork the repository
2. Clone your fork
3. Create a virtual environment
4. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
5. Create a feature branch
6. Make your changes
7. Run security checks:
   ```bash
   pip-audit
   black --check .
   flake8 .
   ```
8. Submit a pull request

### Code Quality

Before submitting changes:

```bash
# Format code
black .

# Lint code
flake8 .
pylint *.py

# Security check
pip-audit
```

## License

[Specify your license here]

## Contact

[Add contact information or contribution guidelines]

## Acknowledgments

- Dataset source: [Specify data source]
- Built with scikit-learn, pandas, and other open-source libraries

---

**Last Updated:** 2024
**Python Version:** 3.8.18
**Project Status:** Active Development
