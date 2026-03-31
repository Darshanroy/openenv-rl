import os
import sys

def check_file(path, description):
    if os.path.exists(path):
        print(f"[PASS] {description} found at {path}")
        return True
    else:
        print(f"[FAIL] {description} NOT FOUND at {path}")
        return False

def check_inference_script():
    path = "inference.py"
    if not os.path.exists(path):
        print("[FAIL] inference.py not found in root.")
        return False
    
    with open(path, "r") as f:
        content = f.read()
        
    checks = {
        "OpenAI Client": "from openai import OpenAI" in content or "import openai" in content,
        "API_BASE_URL": "API_BASE_URL" in content and "os.getenv" in content,
        "MODEL_NAME": "MODEL_NAME" in content and "os.getenv" in content,
        "HF_TOKEN": "HF_TOKEN" in content and "os.getenv" in content,
    }
    
    all_pass = True
    for name, result in checks.items():
        if result:
            print(f"[PASS] inference.py uses {name}")
        else:
            print(f"[FAIL] inference.py missing {name} logic")
            all_pass = False
    return all_pass

def main():
    print("="*60)
    print("OPENENV SUBMISSION VALIDATOR")
    print("="*60)
    
    results = [
        check_file("inference.py", "Final Baseline Script"),
        check_file("Dockerfile", "Deployment Container"),
        check_file("openenv.yaml", "Environment Spec (Root)"),
        check_file("README.md", "Documentation"),
        check_file("requirements.txt", "Dependencies"),
        check_inference_script()
    ]
    
    print("="*60)
    if all(results):
        print("[SUCCESS] STATUS: READY FOR SUBMISSION!")
        print("Everything looks 100% compliant with the checklist.")
    else:
        print("[WARNING] STATUS: NOT READY.")
        print("Please fix the failures above before submitting.")
    print("="*60)

if __name__ == "__main__":
    main()
