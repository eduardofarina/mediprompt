#!/usr/bin/env python
"""
Script to validate all clinical reasoning patterns against the defined schema.
"""

import os
import sys
import json
import jsonschema
from typing import Dict, Any, List, Tuple

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Define the schema for clinical reasoning patterns
PATTERN_SCHEMA = {
    "type": "object",
    "required": ["id", "trigger", "strategy", "evidenceBase", "cautions", "example"],
    "properties": {
        "id": {"type": "string", "pattern": "^CRP-\\d{3}$"},
        "trigger": {"type": "string", "minLength": 10},
        "strategy": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1
        },
        "evidenceBase": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["source", "citation", "relevance"],
                "properties": {
                    "source": {"type": "string"},
                    "citation": {"type": "string"},
                    "relevance": {"type": "string"}
                }
            }
        },
        "cautions": {
            "type": "array",
            "items": {"type": "string"}
        },
        "example": {
            "type": "object",
            "required": ["presentation", "application", "outcome"],
            "properties": {
                "presentation": {"type": "string"},
                "application": {"type": "string"},
                "outcome": {"type": "string"}
            }
        },
        "dateAdded": {"type": "string", "format": "date-time"},
        "lastUpdated": {"type": "string", "format": "date-time"}
    }
}

def load_patterns(patterns_dir: str) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Load all clinical reasoning patterns from the given directory.
    
    Args:
        patterns_dir: Directory containing pattern JSON files
        
    Returns:
        List of tuples with (filename, pattern_data)
    """
    patterns = []
    
    if not os.path.isdir(patterns_dir):
        print(f"Error: Patterns directory '{patterns_dir}' does not exist")
        return patterns
    
    for filename in os.listdir(patterns_dir):
        if filename.endswith(".json"):
            try:
                with open(os.path.join(patterns_dir, filename), "r") as f:
                    pattern_data = json.load(f)
                    patterns.append((filename, pattern_data))
            except Exception as e:
                print(f"Error loading pattern from {filename}: {e}")
    
    return patterns

def validate_pattern(pattern: Dict[str, Any], filename: str) -> List[str]:
    """
    Validate a clinical reasoning pattern against the schema.
    
    Args:
        pattern: Pattern data
        filename: Source filename for error reporting
        
    Returns:
        List of validation error messages, empty if valid
    """
    errors = []
    
    try:
        jsonschema.validate(instance=pattern, schema=PATTERN_SCHEMA)
    except jsonschema.exceptions.ValidationError as e:
        errors.append(f"{filename}: {e.message}")
    
    # Additional custom validations
    if "id" in pattern and not pattern["id"].startswith("CRP-"):
        errors.append(f"{filename}: Pattern ID must start with 'CRP-'")
    
    if "strategy" in pattern and len(pattern["strategy"]) < 3:
        errors.append(f"{filename}: Strategy should have at least 3 steps for comprehensive reasoning")
    
    if "cautions" in pattern and len(pattern["cautions"]) < 2:
        errors.append(f"{filename}: Should include at least 2 cautions for safe application")
    
    return errors

def main():
    """Main function to validate clinical reasoning patterns."""
    # Get the patterns directory path
    patterns_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "patterns"))
    
    print(f"Validating patterns in {patterns_dir}...")
    
    # Load patterns
    patterns = load_patterns(patterns_dir)
    
    if not patterns:
        print("No patterns found!")
        sys.exit(1)
    
    print(f"Found {len(patterns)} patterns")
    
    # Validate each pattern
    all_errors = []
    valid_count = 0
    
    for filename, pattern in patterns:
        errors = validate_pattern(pattern, filename)
        
        if errors:
            all_errors.extend(errors)
        else:
            valid_count += 1
            print(f"✓ {filename} is valid")
    
    # Print validation results
    print("\nValidation Results:")
    print(f"- Total patterns: {len(patterns)}")
    print(f"- Valid patterns: {valid_count}")
    print(f"- Invalid patterns: {len(patterns) - valid_count}")
    
    if all_errors:
        print("\nErrors found:")
        for error in all_errors:
            print(f"- {error}")
        sys.exit(1)
    else:
        print("\nAll patterns are valid!")
        sys.exit(0)

if __name__ == "__main__":
    main() 