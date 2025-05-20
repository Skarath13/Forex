#!/usr/bin/env python3
"""
Fix Pandas Deprecation Warnings

This script fixes the Series.__getitem__ deprecation warnings in the adaptive_strategy.py file.
"""

import os
import re

def fix_pandas_deprecation_warnings():
    """
    Update Series indexing from [-1] to .iloc[-1] format to address pandas deprecation warnings
    """
    print("Fixing pandas deprecation warnings in adaptive_strategy.py...")
    
    strategy_file = "/Users/dylan/Desktop/Forex/src/strategies/adaptive_strategy.py"
    
    # Read the file
    with open(strategy_file, 'r') as f:
        content = f.read()
    
    # Define replacement patterns
    # This regex will find Series indexing patterns like:
    # series_name['column_name'][-1] and replace with series_name['column_name'].iloc[-1]
    pattern = r"(\w+\[['\"][^'\"]+['\"]]\[)(-\d+)(\])"
    
    # Replace the pattern
    updated_content = re.sub(pattern, r"\1.iloc\2\3", content)
    
    # A more specific pattern for ranges (e.g., [-20:])
    range_pattern = r"(\w+\[['\"][^'\"]+['\"]]\[)(-\d+)(:\])"
    updated_content = re.sub(range_pattern, r"\1.iloc\2\3", updated_content)
    
    # Check if any changes were made
    if content == updated_content:
        print("No changes were required or pattern didn't match.")
        return
    
    # Make a backup of the original file
    backup_file = strategy_file + ".bak"
    os.rename(strategy_file, backup_file)
    
    # Write the updated content
    with open(strategy_file, 'w') as f:
        f.write(updated_content)
    
    print(f"Successfully updated {strategy_file}")
    print(f"Backup saved as {backup_file}")

if __name__ == "__main__":
    fix_pandas_deprecation_warnings()