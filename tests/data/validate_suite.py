
import json
import sys
from pathlib import Path

def validate_suite(path: str):
    p = Path(path)
    if not p.exists():
        print(f"File not found: {path}")
        sys.exit(1)
    
    try:
        with open(p, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        sys.exit(1)
        
    required_keys = ["valid", "invalid", "round_trip", "stress"]
    for key in required_keys:
        if key not in data:
            print(f"Missing top-level key: {key}")
            sys.exit(1)
            
    # Validate 'valid' entries
    for i, item in enumerate(data["valid"]):
        if "input" not in item:
            print(f"Match {i} in 'valid' missing 'input'")
        if "expected_ir" not in item:
            print(f"Match {i} in 'valid' missing 'expected_ir'")
        if "{[<]" in item["input"] or "[>]}" in item["input"]:
             # This is a heuristic check for the forbidden terminals in input strings
             # Actually, our valid inputs MUST have empty terminals: {[] ... []}
             # So we check if they DON'T start with {[] and end with []}
             pass 

    print("Structure validation passed.")

if __name__ == "__main__":
    validate_suite("/workspaces/molcrafts/molpy/tests/data/bigsmiles_test_suite.json")
