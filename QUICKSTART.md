# Quick Start Guide

Get up and running with the Electricity Price Prediction project in 5 minutes.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- git (for cloning the repository)

## Step-by-Step Setup

### 1. Clone the Repository (if applicable)

```bash
git clone <repository-url>
cd <project-directory>
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**On Linux/macOS:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### 4. Install Dependencies

**For running the analysis (recommended for most users):**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**For development (includes Jupyter, testing, security tools):**
```bash
pip install --upgrade pip
pip install -r requirements-dev.txt
```

### 5. Verify Installation

```bash
python -c "import pandas, numpy, sklearn, matplotlib, seaborn, statsmodels; print('✅ Installation successful!')"
```

## Running the Project

### Option 1: Run Python Script

```bash
python "Electricity Data.py"
```

### Option 2: Use Jupyter Notebook

```bash
# Start Jupyter Notebook
jupyter notebook

# Open "Electricity Data.ipynb" from the browser
```

### Option 3: Run Hyperparameter Tuning

```bash
python hyperparameter_tuning.py
```

**Note:** This may take 1-2 hours depending on your hardware.

## Security Check (Optional but Recommended)

If you installed development dependencies:

```bash
# Check for security vulnerabilities
pip-audit

# Alternative security check
safety check
```

## Troubleshooting

### Problem: "python: command not found"

Try `python3` instead:
```bash
python3 -m venv venv
```

### Problem: "No module named 'X'"

Make sure virtual environment is activated and dependencies are installed:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Problem: "Permission denied"

Don't use `sudo`. Use virtual environments:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Problem: Jupyter kernel not found

Install the kernel:
```bash
pip install ipykernel
python -m ipykernel install --user --name=electricity-prediction
```

## What's Next?

- Read [README.md](README.md) for detailed documentation
- Check [DEPENDENCY_MANAGEMENT.md](DEPENDENCY_MANAGEMENT.md) for dependency updates
- Review [HYPERPARAMETER_TUNING_README.md](HYPERPARAMETER_TUNING_README.md) for model optimization

## Need Help?

1. Check [README.md](README.md) Troubleshooting section
2. Review [DEPENDENCY_MANAGEMENT.md](DEPENDENCY_MANAGEMENT.md)
3. Verify all dependencies are installed: `pip list`
4. Check Python version: `python --version` (should be 3.8+)

---

**Total setup time:** ~5 minutes  
**Dataset required:** electricity.csv (included)  
**Disk space needed:** ~500MB for dependencies
