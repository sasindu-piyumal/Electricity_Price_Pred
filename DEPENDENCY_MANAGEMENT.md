# Dependency Management Guide

This document provides comprehensive guidance on managing dependencies for the Electricity Price Prediction project.

## Table of Contents

- [Overview](#overview)
- [Dependency Files](#dependency-files)
- [Installation Guide](#installation-guide)
- [Security Scanning](#security-scanning)
- [Update Procedures](#update-procedures)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Overview

This project uses pinned dependency versions to ensure reproducible builds and prevent supply chain vulnerabilities. All dependencies are managed through pip and requirements files.

### Key Principles

1. **Version Pinning**: All dependencies use exact version pins (e.g., `package==X.Y.Z`)
2. **Separation of Concerns**: Production and development dependencies are separated
3. **Security First**: Regular vulnerability scanning with pip-audit and safety
4. **Reproducibility**: Consistent builds across all environments

## Dependency Files

### `requirements.txt` - Production Dependencies

Core dependencies required to run the analysis and prediction models:

```
pandas==2.0.3          # Data manipulation and analysis
numpy==1.24.3          # Numerical computing
scikit-learn==1.3.0    # Machine learning
statsmodels==0.14.0    # Statistical modeling
matplotlib==3.7.2      # Plotting and visualization
seaborn==0.12.2        # Statistical visualization
```

**Installation:**
```bash
pip install -r requirements.txt
```

### `requirements-dev.txt` - Development Dependencies

Additional tools for development, testing, and security:

```
-r requirements.txt    # Includes all production dependencies
jupyter==1.0.0         # Interactive notebooks
notebook==7.0.0        # Jupyter notebook server
ipython==8.14.0        # Enhanced Python shell
pytest==7.4.0          # Testing framework
black==23.7.0          # Code formatter
flake8==6.1.0          # Linting
pip-audit==2.6.1       # Security scanning
safety==2.3.5          # Vulnerability checking
```

**Installation:**
```bash
pip install -r requirements-dev.txt
```

### `runtime.txt` - Python Version

Specifies the required Python version for deployment platforms (e.g., Heroku):

```
python-3.8.18
```

### `.python-version` - pyenv Configuration

Specifies the Python version for pyenv users:

```
3.8.18
```

## Installation Guide

### Fresh Environment Setup

#### Option 1: Using venv (Standard)

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
# or for development
pip install -r requirements-dev.txt
```

#### Option 2: Using conda

```bash
# Create environment
conda create -n electricity-prediction python=3.8

# Activate
conda activate electricity-prediction

# Install pip dependencies
pip install -r requirements.txt
```

#### Option 3: Using pyenv + venv

```bash
# Install specific Python version
pyenv install 3.8.18

# Set local version (uses .python-version)
pyenv local 3.8.18

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Verification

After installation, verify all packages are correctly installed:

```bash
# Check installed packages
pip list

# Verify key imports
python -c "import pandas, numpy, sklearn, matplotlib, seaborn, statsmodels; print('✓ All dependencies installed')"

# Check for dependency conflicts
pip check
```

## Security Scanning

### Using pip-audit (Recommended)

pip-audit is a tool that scans Python packages for known security vulnerabilities.

#### Basic Usage

```bash
# Install pip-audit
pip install pip-audit

# Run basic scan
pip-audit

# Scan with descriptions
pip-audit --desc

# Generate JSON report
pip-audit --format json --output security-report.json

# Check specific requirements file
pip-audit -r requirements.txt
```

#### Advanced Usage

```bash
# Scan with CVE details
pip-audit --desc --format cyclonedx-json --output sbom.json

# Fix vulnerabilities automatically (use with caution)
pip-audit --fix

# Strict mode (fail on any vulnerability)
pip-audit --strict

# Ignore specific vulnerabilities (when fix not available)
pip-audit --ignore-vuln GHSA-xxxx-xxxx-xxxx
```

### Using safety

safety checks Python dependencies for known security vulnerabilities.

```bash
# Install safety
pip install safety

# Basic check
safety check

# Full report
safety check --full-report

# JSON output
safety check --json

# Check specific file
safety check -r requirements.txt

# Generate detailed report
safety check --output text --save-json safety-report.json
```

### Automated Scanning (CI/CD)

The project includes a GitHub Actions workflow (`.github/workflows/security-scan.yml`) that:

1. Runs on every push to main/master
2. Runs on all pull requests
3. Runs weekly (every Monday)
4. Can be triggered manually

**Workflow features:**
- Dual scanning with pip-audit and safety
- JSON report generation
- Artifact upload for historical tracking
- Security summary in GitHub Actions output

### Manual Scanning Schedule

Recommended scanning frequency:

- **Daily**: During active development
- **Weekly**: For stable projects (automated via GitHub Actions)
- **Before deployment**: Always scan before production releases
- **After dependency updates**: Immediately after updating any package

## Update Procedures

### Security Updates (Patch Versions)

When a security vulnerability is discovered:

1. **Identify the vulnerable package:**
   ```bash
   pip-audit --desc
   ```

2. **Check for available updates:**
   ```bash
   pip index versions package-name
   ```

3. **Update the package:**
   ```bash
   pip install --upgrade package-name==X.Y.Z
   ```

4. **Update requirements file:**
   ```bash
   # For requirements.txt
   pip freeze | grep package-name
   # Manually update requirements.txt with new version
   ```

5. **Test the application:**
   ```bash
   python "Electricity Data.py"
   python hyperparameter_tuning.py
   ```

6. **Verify the fix:**
   ```bash
   pip-audit
   ```

7. **Commit changes:**
   ```bash
   git add requirements.txt
   git commit -m "Security update: package-name to X.Y.Z (fixes CVE-YYYY-XXXXX)"
   ```

### Minor Version Updates

To update packages to newer minor versions (e.g., 2.0.3 → 2.1.0):

1. **Create a test branch:**
   ```bash
   git checkout -b update-dependencies
   ```

2. **Update one package at a time:**
   ```bash
   pip install --upgrade package-name
   ```

3. **Run all tests and scripts:**
   ```bash
   python "Electricity Data.py"
   python hyperparameter_tuning.py
   python analyze_tuning_results.py
   ```

4. **Check for deprecation warnings:**
   ```bash
   python -W default "Electricity Data.py"
   ```

5. **Update requirements:**
   ```bash
   pip freeze > requirements-new.txt
   # Review and merge into requirements.txt
   ```

6. **Run security scan:**
   ```bash
   pip-audit
   ```

7. **Commit and test thoroughly:**
   ```bash
   git add requirements.txt
   git commit -m "Update package-name to X.Y.Z"
   ```

### Major Version Updates

Major version updates (e.g., pandas 2.0 → 3.0) require careful testing:

1. **Review changelog** for breaking changes
2. **Create isolated test environment**
3. **Update and test incrementally**
4. **Update code for API changes**
5. **Run full test suite**
6. **Update documentation**

### Bulk Updates (Use with Caution)

To check for all available updates:

```bash
# List outdated packages
pip list --outdated

# Using pip-review (install first: pip install pip-review)
pip-review --local --interactive
```

**Warning**: Never bulk update in production without thorough testing.

## Troubleshooting

### Common Issues and Solutions

#### Issue: Dependency Conflicts

```
ERROR: package-a requires package-b>=2.0, but you have package-b 1.5
```

**Solution:**
```bash
# Check dependency tree
pip install pipdeptree
pipdeptree -p package-a

# Resolve conflicts
pip install package-b>=2.0
pip check
```

#### Issue: No Matching Distribution

```
ERROR: Could not find a version that satisfies the requirement package==X.Y.Z
```

**Solution:**
```bash
# Check available versions
pip index versions package

# Update to available version
pip install package==X.Y.W  # where W is available
```

#### Issue: pip-audit Fails

```
ERROR: Failed to scan dependencies
```

**Solution:**
```bash
# Update pip and pip-audit
pip install --upgrade pip pip-audit

# Clear cache
pip cache purge

# Retry scan
pip-audit --cache-dir /tmp/pip-audit-cache
```

#### Issue: Installation Fails on Specific Package

**Solution for numpy/scikit-learn compilation issues:**
```bash
# Install build dependencies (Ubuntu/Debian)
sudo apt-get install python3-dev build-essential

# Install build dependencies (macOS)
brew install openblas

# Use pre-built wheels
pip install --only-binary :all: package-name
```

### Clean Reinstall

If dependencies are completely broken:

```bash
# Deactivate and remove virtual environment
deactivate
rm -rf venv

# Create fresh environment
python -m venv venv
source venv/bin/activate

# Install from scratch
pip install --upgrade pip
pip install -r requirements.txt
```

## Best Practices

### 1. Version Pinning Strategy

**DO:**
- ✅ Pin exact versions for production (`package==X.Y.Z`)
- ✅ Document reason for specific versions
- ✅ Review dependencies before adding new ones

**DON'T:**
- ❌ Use version ranges in production (`package>=X.Y`)
- ❌ Install packages without version pins
- ❌ Add unnecessary dependencies

### 2. Security Practices

**DO:**
- ✅ Run pip-audit before every deployment
- ✅ Subscribe to security advisories
- ✅ Update security patches promptly
- ✅ Use virtual environments always
- ✅ Review dependency licenses

**DON'T:**
- ❌ Ignore security warnings
- ❌ Install packages system-wide
- ❌ Use packages from untrusted sources
- ❌ Disable security scans

### 3. Update Practices

**DO:**
- ✅ Test updates in isolated environment first
- ✅ Update one package at a time
- ✅ Read changelogs before updating
- ✅ Keep backups of working requirements
- ✅ Document update reasons in commits

**DON'T:**
- ❌ Update all packages at once
- ❌ Skip testing after updates
- ❌ Update in production directly
- ❌ Ignore breaking changes

### 4. Documentation

**DO:**
- ✅ Keep requirements files up-to-date
- ✅ Document Python version requirements
- ✅ Comment complex dependency relationships
- ✅ Track security scan results

**DON'T:**
- ❌ Forget to commit updated requirements
- ❌ Leave outdated documentation
- ❌ Skip documenting breaking changes

### 5. Environment Management

**DO:**
- ✅ Use separate environments for each project
- ✅ Recreate environments periodically
- ✅ Document environment setup steps
- ✅ Use .gitignore for venv directories

**DON'T:**
- ❌ Share environments between projects
- ❌ Commit virtual environment to git
- ❌ Install packages globally
- ❌ Mix conda and pip haphazardly

## Additional Tools

### pip-tools

For even more reproducible builds:

```bash
# Install pip-tools
pip install pip-tools

# Create requirements.in with unpinned versions
# Then compile to requirements.txt with full dependency tree
pip-compile requirements.in

# Upgrade all dependencies
pip-compile --upgrade requirements.in
```

### Poetry (Alternative)

Modern dependency management:

```bash
# Install poetry
pip install poetry

# Initialize project
poetry init

# Add dependencies
poetry add pandas numpy scikit-learn

# Install
poetry install
```

### pipdeptree

Visualize dependency tree:

```bash
pip install pipdeptree
pipdeptree
pipdeptree -p scikit-learn
```

## References

- [pip Documentation](https://pip.pypa.io/)
- [pip-audit Documentation](https://pypi.org/project/pip-audit/)
- [Python Packaging Guide](https://packaging.python.org/)
- [PyPA Security](https://www.pypa.io/en/latest/specifications/security/)
- [National Vulnerability Database](https://nvd.nist.gov/)

## Support

For issues related to dependency management:

1. Check this document first
2. Review pip-audit and safety reports
3. Check package documentation
4. Search for similar issues on GitHub/Stack Overflow
5. Create an issue with full error details

---

**Last Updated:** 2024  
**Maintained by:** [Project Team]
