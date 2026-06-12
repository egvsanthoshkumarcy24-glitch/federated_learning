import os
import subprocess

def run_script(script_path):
    print(f"\n{'='*50}\nRunning {script_path}...\n{'='*50}")
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_path}:\n{result.stderr}")
    else:
        print(result.stdout)
        print(f"Successfully completed {script_path}.")

if __name__ == "__main__":
    scripts = [
        "data/generate_synthetic_data.py",
        "data/clean.py",
        "data/partition.py",
        "data/inject_anomalies.py",
        "models/train.py",
        "eval/centralized_baseline.py",
        "run_federated.py",
        "eval/evaluate.py"
    ]
    
    for script in scripts:
        run_script(script)
