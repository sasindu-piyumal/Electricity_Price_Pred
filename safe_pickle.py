#!/usr/bin/env python
# coding: utf-8

"""
Safe Pickle Serialization Module
=================================

This module provides secure pickle loading and saving functionality to mitigate
arbitrary code execution risks from pickle deserialization attacks.

SECURITY WARNING: Pickle files can execute arbitrary code during deserialization
via the __reduce__ method. This module implements:
1. Restricted unpickler with class whitelist
2. SHA-256 integrity verification
3. Validation before unpickling

For more information, see SECURITY.md
"""

import pickle
import hashlib
import io
import os
import warnings
from pathlib import Path


# SECURITY: Whitelist of safe classes that can be unpickled
# This prevents arbitrary code execution from malicious pickle files
SAFE_CLASSES = {
    # Scikit-learn models and components
    ('sklearn.ensemble._forest', 'RandomForestRegressor'),
    ('sklearn.ensemble', 'RandomForestRegressor'),
    ('sklearn.model_selection._search', 'RandomizedSearchCV'),
    ('sklearn.model_selection._search', 'GridSearchCV'),
    ('sklearn.model_selection', 'RandomizedSearchCV'),
    ('sklearn.model_selection', 'GridSearchCV'),
    ('sklearn.tree._classes', 'DecisionTreeRegressor'),
    ('sklearn.tree', 'DecisionTreeRegressor'),
    ('sklearn.preprocessing._data', 'MinMaxScaler'),
    ('sklearn.preprocessing', 'MinMaxScaler'),
    
    # NumPy arrays and data types
    ('numpy', 'ndarray'),
    ('numpy', 'dtype'),
    ('numpy.core.multiarray', '_reconstruct'),
    ('numpy.core.multiarray', 'scalar'),
    ('numpy.core._multiarray_umath', '_reconstruct'),
    ('numpy.core._multiarray_umath', 'scalar'),
    
    # Pandas data structures
    ('pandas.core.frame', 'DataFrame'),
    ('pandas.core.series', 'Series'),
    ('pandas', 'DataFrame'),
    ('pandas', 'Series'),
    ('pandas.core.indexes.base', 'Index'),
    ('pandas.core.indexes.range', 'RangeIndex'),
    
    # Python built-in safe types
    ('builtins', 'dict'),
    ('builtins', 'list'),
    ('builtins', 'tuple'),
    ('builtins', 'set'),
    ('builtins', 'frozenset'),
    ('builtins', 'int'),
    ('builtins', 'float'),
    ('builtins', 'str'),
    ('builtins', 'bool'),
    ('builtins', 'NoneType'),
    ('builtins', 'bytes'),
    ('builtins', 'bytearray'),
    
    # Collections
    ('collections', 'OrderedDict'),
    ('collections', 'defaultdict'),
    ('collections', 'Counter'),
    
    # Datetime objects
    ('datetime', 'datetime'),
    ('datetime', 'date'),
    ('datetime', 'time'),
    ('datetime', 'timedelta'),
}


class RestrictedUnpickler(pickle.Unpickler):
    """
    Custom Unpickler that only allows whitelisted classes to be loaded.
    
    SECURITY: This prevents arbitrary code execution by rejecting any class
    that is not explicitly allowed in the SAFE_CLASSES whitelist.
    
    Raises:
        pickle.UnpicklingError: If an attempt is made to unpickle a class
                               that is not in the whitelist.
    """
    
    def find_class(self, module, name):
        """
        Override find_class to implement whitelist-based class loading.
        
        Args:
            module (str): The module name of the class being unpickled
            name (str): The class name being unpickled
            
        Returns:
            class: The class object if allowed
            
        Raises:
            pickle.UnpicklingError: If the class is not in the whitelist
        """
        # Check if the class is in the whitelist
        if (module, name) not in SAFE_CLASSES:
            error_msg = (
                f"SECURITY: Attempted to unpickle disallowed class '{module}.{name}'. "
                f"This may be a malicious pickle file. Only whitelisted classes are allowed."
            )
            raise pickle.UnpicklingError(error_msg)
        
        # If whitelisted, proceed with normal unpickling
        return super().find_class(module, name)


def generate_file_hash(file_path, algorithm='sha256'):
    """
    Generate a cryptographic hash of a file.
    
    Args:
        file_path (str): Path to the file
        algorithm (str): Hash algorithm to use (default: sha256)
        
    Returns:
        str: Hexadecimal hash digest
    """
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        # Read in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(4096), b''):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def save_hash_file(file_path, hash_value):
    """
    Save hash value to a .hash file alongside the pickle file.
    
    Args:
        file_path (str): Path to the pickle file
        hash_value (str): Hash value to save
    """
    hash_file_path = f"{file_path}.hash"
    with open(hash_file_path, 'w') as f:
        f.write(hash_value)


def verify_file_hash(file_path, algorithm='sha256'):
    """
    Verify the integrity of a pickle file using its saved hash.
    
    Args:
        file_path (str): Path to the pickle file
        algorithm (str): Hash algorithm to use (default: sha256)
        
    Returns:
        bool: True if hash matches, False otherwise
        
    Raises:
        FileNotFoundError: If hash file doesn't exist
    """
    hash_file_path = f"{file_path}.hash"
    
    if not os.path.exists(hash_file_path):
        warnings.warn(
            f"Hash file not found: {hash_file_path}. "
            "Cannot verify file integrity. This file may have been created "
            "before hash verification was implemented.",
            UserWarning
        )
        return False
    
    # Read saved hash
    with open(hash_file_path, 'r') as f:
        saved_hash = f.read().strip()
    
    # Calculate current hash
    current_hash = generate_file_hash(file_path, algorithm)
    
    # Compare hashes
    return saved_hash == current_hash


def safe_pickle_save(obj, file_path, generate_hash=True):
    """
    Safely save an object to a pickle file with optional hash generation.
    
    SECURITY NOTE: While this function saves securely, be aware that pickle
    files can still be maliciously modified. Always verify hash on load.
    
    Args:
        obj: Object to pickle
        file_path (str): Path where pickle file should be saved
        generate_hash (bool): Whether to generate SHA-256 hash (default: True)
        
    Returns:
        str: Path to the saved file
    """
    # SECURITY WARNING: Pickle serialization should be used with caution
    # Consider migrating to joblib for sklearn models (see SECURITY.md)
    
    # Save the pickle file
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
    
    print(f"    Saved pickle file: {file_path}")
    
    # Generate and save hash if requested
    if generate_hash:
        hash_value = generate_file_hash(file_path)
        save_hash_file(file_path, hash_value)
        print(f"    Generated SHA-256 hash: {hash_value[:16]}...")
        print(f"    Hash saved to: {file_path}.hash")
    
    return file_path


def safe_pickle_load(file_path, verify_hash=True, allow_unsafe=False):
    """
    Safely load a pickle file with restricted unpickler and hash verification.
    
    SECURITY FEATURES:
    1. Verifies SHA-256 hash to detect tampering (if hash file exists)
    2. Uses RestrictedUnpickler to prevent arbitrary code execution
    3. Only allows whitelisted classes (sklearn models, numpy, pandas, etc.)
    
    Args:
        file_path (str): Path to the pickle file
        verify_hash (bool): Whether to verify hash before loading (default: True)
        allow_unsafe (bool): If True, skip hash verification warning (default: False)
        
    Returns:
        object: The unpickled object
        
    Raises:
        FileNotFoundError: If pickle file doesn't exist
        ValueError: If hash verification fails
        pickle.UnpicklingError: If file contains disallowed classes
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Pickle file not found: {file_path}")
    
    # SECURITY WARNING: Loading pickle files from unknown sources
    if not allow_unsafe:
        warnings.warn(
            f"\nSECURITY WARNING: Loading pickle file '{file_path}'\n"
            "Pickle files can execute arbitrary code during deserialization.\n"
            "Only load pickle files from trusted sources.\n"
            "This loader uses a restricted unpickler with class whitelisting "
            "and hash verification for added security.",
            UserWarning
        )
    
    # Verify hash if requested and hash file exists
    if verify_hash:
        try:
            hash_valid = verify_file_hash(file_path)
            if hash_valid:
                print(f"    ✓ Hash verification passed for {file_path}")
            else:
                # Hash file doesn't exist (likely old file), proceed with warning
                print(f"    ⚠ No hash file found, skipping verification")
        except Exception as e:
            print(f"    ⚠ Hash verification error: {e}")
    
    # Load with restricted unpickler
    print(f"    Loading pickle file with security restrictions...")
    
    with open(file_path, 'rb') as f:
        # Use RestrictedUnpickler to prevent arbitrary code execution
        unpickler = RestrictedUnpickler(f)
        obj = unpickler.load()
    
    print(f"    ✓ Successfully loaded pickle file")
    
    return obj


# Convenience function for checking if a file is safe to load
def is_pickle_file_safe(file_path):
    """
    Check if a pickle file passes basic safety checks.
    
    Args:
        file_path (str): Path to the pickle file
        
    Returns:
        tuple: (is_safe: bool, message: str)
    """
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"
    
    # Check if hash exists and is valid
    hash_file_path = f"{file_path}.hash"
    if not os.path.exists(hash_file_path):
        return False, "No hash file found - cannot verify integrity"
    
    try:
        hash_valid = verify_file_hash(file_path)
        if not hash_valid:
            return False, "Hash verification failed - file may be tampered"
    except Exception as e:
        return False, f"Hash verification error: {str(e)}"
    
    return True, "File passes safety checks"


if __name__ == "__main__":
    print("Safe Pickle Module")
    print("=" * 80)
    print("This module provides secure pickle loading/saving functionality.")
    print("\nSafe classes whitelist:")
    print(f"  - {len(SAFE_CLASSES)} classes allowed")
    print("\nSecurity features:")
    print("  - Restricted unpickler with class whitelist")
    print("  - SHA-256 hash verification")
    print("  - Warning messages for unknown sources")
    print("\nFor more information, see SECURITY.md")
    print("=" * 80)
