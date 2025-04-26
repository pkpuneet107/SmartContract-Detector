import os
import json
import subprocess
from tqdm import tqdm

def extract_ast_from_sol(source_path, output_path):
    try:
        result = subprocess.run(
            ["solc", "--ast-compact-json", source_path],
            capture_output=True,
            text=True,
            check=True
        )
        with open(output_path, "w") as f:
            f.write(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to parse: {source_path}")
        return False

def batch_extract_asts(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sol_files = [f for f in os.listdir(input_dir) if f.endswith(".sol")]

    for filename in tqdm(sol_files, desc="Extracting ASTs"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".sol", "_ast.json"))
        extract_ast_from_sol(input_path, output_path)

if __name__ == "__main__":
    input_folder = "data/raw/secure"  # Update path as needed
    output_folder = "data/graphs/secure_asts"
    batch_extract_asts(input_folder, output_folder)
