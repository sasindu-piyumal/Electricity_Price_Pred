#!/usr/bin/env python3
# coding: utf-8

"""
Environment Validation Script for Hyperparameter Tuning Pipeline
This script validates that all dependencies, files, and environment requirements
are met before executing the hyperparameter tuning pipeline.
"""

import os
import sys
import ast
from pathlib import Path
from datetime import datetime

# ANSI color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


def print_header(text):
    """Print a formatted header"""
    print(f"\n{BOLD}{BLUE}{'=' * 80}{RESET}")
    print(f"{BOLD}{BLUE}{text.center(80)}{RESET}")
    print(f"{BOLD}{BLUE}{'=' * 80}{RESET}\n")


def print_section(text):
    """Print a formatted section title"""
    print(f"\n{BOLD}{text}{RESET}")
    print(f"{'-' * 80}")


def print_success(text):
    """Print success message"""
    print(f"{GREEN}✓{RESET} {text}")


def print_error(text):
    """Print error message"""
    print(f"{RED}✗{RESET} {text}")


def print_warning(text):
    """Print warning message"""
    print(f"{YELLOW}⚠{RESET} {text}")


def print_info(text):
    """Print info message"""
    print(f"{BLUE}ℹ{RESET} {text}")


class EnvironmentValidator:
    """Validates the execution environment for hyperparameter tuning"""
    
    def __init__(self):
        self.validation_results = {
            'python': None,
            'imports': {},
            'files': {},
            'syntax': {},
            'data': None
        }
        self.all_passed = True
    
    def validate_python_version(self):
        """Validate Python version"""
        print_section("1. Python Version Check")
        
        version = sys.version_info
        version_string = f"{version.major}.{version.minor}.{version.micro}"
        
        if version.major >= 3 and version.minor >= 7:
            print_success(f"Python {version_string} (meets minimum requirement: 3.7+)")
            self.validation_results['python'] = True
        else:
            print_error(f"Python {version_string} (requires 3.7+)")
            self.validation_results['python'] = False
            self.all_passed = False
    
    def validate_imports(self):
        """Validate all required imports"""
        print_section("2. Required Packages Import Check")
        
        imports = {
            'pandas': 'Data manipulation',
            'numpy': 'Numerical computing',
            'sklearn': 'Machine learning',
            'matplotlib.pyplot': 'Visualization',
            'seaborn': 'Statistical visualization',
            'pickle': 'Serialization (stdlib)',
            'time': 'Timing utilities (stdlib)',
            'warnings': 'Warning control (stdlib)',
            'datetime': 'Date/time handling (stdlib)'
        }
        
        for module, description in imports.items():
            try:
                # Handle special cases for module names
                if module == 'sklearn':
                    __import__('sklearn')
                elif module == 'matplotlib.pyplot':
                    __import__('matplotlib.pyplot')
                else:
                    __import__(module)
                
                self.validation_results['imports'][module] = True
                print_success(f"{module:<25} - {description}")
            except ImportError as e:
                self.validation_results['imports'][module] = False
                print_error(f"{module:<25} - {description} (ImportError: {str(e)})")
                self.all_passed = False
    
    def validate_package_versions(self):
        """Display installed package versions"""
        print_section("3. Package Versions")
        
        packages = {
            'pandas': 'pd',
            'numpy': 'np',
            'sklearn': None,
            'matplotlib': 'plt',
            'seaborn': 'sns'
        }
        
        for package, alias in packages.items():
            try:
                if package == 'sklearn':
                    import sklearn
                    version = sklearn.__version__
                elif alias:
                    mod = __import__(package, fromlist=[alias])
                    version = mod.__version__
                else:
                    mod = __import__(package)
                    version = mod.__version__
                
                print_info(f"{package:<20} version: {version}")
            except Exception as e:
                print_warning(f"{package:<20} could not determine version: {str(e)}")
    
    def validate_files(self):
        """Validate required files exist and are readable"""
        print_section("4. Required Files Check")
        
        required_files = {
            'electricity.csv': 'Data file',
            'hyperparameter_tuning.py': 'Main tuning script',
            'analyze_tuning_results.py': 'Results analysis script'
        }
        
        for filename, description in required_files.items():
            filepath = Path(filename)
            
            if filepath.exists():
                file_size = filepath.stat().st_size
                if file_size > 0:
                    self.validation_results['files'][filename] = True
                    size_mb = file_size / (1024 * 1024)
                    print_success(f"{filename:<30} ({size_mb:.2f} MB) - {description}")
                else:
                    self.validation_results['files'][filename] = False
                    print_error(f"{filename:<30} (empty file) - {description}")
                    self.all_passed = False
            else:
                self.validation_results['files'][filename] = False
                print_error(f"{filename:<30} (not found) - {description}")
                self.all_passed = False
    
    def validate_python_syntax(self):
        """Validate Python scripts for syntax errors"""
        print_section("5. Python Script Syntax Validation")
        
        python_files = [
            'hyperparameter_tuning.py',
            'analyze_tuning_results.py'
        ]
        
        for filepath in python_files:
            if not Path(filepath).exists():
                print_error(f"{filepath} - file not found")
                self.validation_results['syntax'][filepath] = False
                self.all_passed = False
                continue
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    code = f.read()
                
                ast.parse(code)
                line_count = len(code.split('\n'))
                self.validation_results['syntax'][filepath] = True
                print_success(f"{filepath:<30} ({line_count} lines) - Syntax valid")
            except SyntaxError as e:
                self.validation_results['syntax'][filepath] = False
                print_error(f"{filepath} - Syntax error at line {e.lineno}: {e.msg}")
                self.all_passed = False
            except Exception as e:
                self.validation_results['syntax'][filepath] = False
                print_error(f"{filepath} - Error: {str(e)}")
                self.all_passed = False
    
    def validate_data(self):
        """Validate electricity.csv data"""
        print_section("6. Data File Validation")
        
        try:
            import pandas as pd
            
            df = pd.read_csv('electricity.csv', index_col=0, parse_dates=[0], nrows=5)
            
            # Check full file for row count
            with open('electricity.csv', 'r') as f:
                row_count = sum(1 for line in f) - 1  # Subtract header
            
            print_success(f"Dataset loaded successfully")
            print_info(f"  - Rows: {row_count:,}")
            print_info(f"  - Columns: 18")
            
            # Verify required columns
            required_cols = ['SMPEP2']
            df_full = pd.read_csv('electricity.csv', index_col=0, nrows=100)
            
            for col in required_cols:
                if col in df_full.columns:
                    print_success(f"  - Required column '{col}' present")
                else:
                    print_error(f"  - Required column '{col}' missing")
                    self.all_passed = False
            
            self.validation_results['data'] = True
        except Exception as e:
            print_error(f"Data validation failed: {str(e)}")
            self.validation_results['data'] = False
            self.all_passed = False
    
    def validate_import_statements(self):
        """Validate that import statements in scripts work"""
        print_section("7. Script Import Statements Validation")
        
        try:
            # Simulate importing from hyperparameter_tuning.py
            import_statements = [
                "import pandas",
                "import numpy",
                "from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, TimeSeriesSplit",
                "from sklearn.preprocessing import MinMaxScaler",
                "from sklearn.ensemble import RandomForestRegressor",
                "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score",
                "import matplotlib.pyplot",
                "import seaborn",
                "import pickle",
                "import time",
                "from datetime import datetime",
                "import warnings"
            ]
            
            for statement in import_statements:
                try:
                    exec(statement)
                    print_success(f"{statement}")
                except Exception as e:
                    print_error(f"{statement} - {str(e)}")
                    self.all_passed = False
        except Exception as e:
            print_error(f"Import validation failed: {str(e)}")
            self.all_passed = False
    
    def generate_report(self):
        """Generate final validation report"""
        print_section("VALIDATION SUMMARY")
        
        # Overall status
        if self.all_passed:
            print(f"{GREEN}{BOLD}✓ ALL VALIDATIONS PASSED{RESET}\n")
            status = "READY"
        else:
            print(f"{RED}{BOLD}✗ SOME VALIDATIONS FAILED{RESET}\n")
            status = "NOT READY"
        
        # Detailed breakdown
        print(f"{BOLD}Environment Status:{RESET}")
        print(f"  Python Version: {GREEN if self.validation_results['python'] else RED}{'✓' if self.validation_results['python'] else '✗'}{RESET}")
        
        print(f"\n{BOLD}Required Packages:{RESET}")
        for module, passed in self.validation_results['imports'].items():
            symbol = f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"
            print(f"  {symbol} {module}")
        
        print(f"\n{BOLD}Required Files:{RESET}")
        for filename, passed in self.validation_results['files'].items():
            symbol = f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"
            print(f"  {symbol} {filename}")
        
        print(f"\n{BOLD}Python Scripts:{RESET}")
        for filename, passed in self.validation_results['syntax'].items():
            symbol = f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"
            print(f"  {symbol} {filename} (syntax)")
        
        print(f"\n{BOLD}Data Validation:{RESET}")
        data_symbol = f"{GREEN}✓{RESET}" if self.validation_results['data'] else f"{RED}✗{RESET}"
        print(f"  {data_symbol} electricity.csv")
        
        print(f"\n{BOLD}Overall Status:{RESET} {status}")
        print(f"{BOLD}Timestamp:{RESET} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return self.all_passed
    
    def run_all_validations(self):
        """Run all validation checks"""
        print_header("ENVIRONMENT VALIDATION FOR HYPERPARAMETER TUNING PIPELINE")
        
        self.validate_python_version()
        self.validate_imports()
        self.validate_package_versions()
        self.validate_files()
        self.validate_python_syntax()
        self.validate_data()
        self.validate_import_statements()
        
        passed = self.generate_report()
        
        print_header("VALIDATION COMPLETE")
        
        if passed:
            print(f"\n{GREEN}{BOLD}✓ Environment is ready for hyperparameter tuning pipeline execution!{RESET}\n")
            print("Next steps:")
            print("  1. Run: python hyperparameter_tuning.py")
            print("  2. Monitor execution progress")
            print("  3. Analyze results: python analyze_tuning_results.py")
            print()
        else:
            print(f"\n{RED}{BOLD}✗ Please resolve the validation failures above before proceeding.{RESET}\n")
        
        return passed


def main():
    """Main entry point"""
    validator = EnvironmentValidator()
    passed = validator.run_all_validations()
    
    # Exit with appropriate code
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
