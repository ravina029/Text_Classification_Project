import os

# Define the folder structure
folders = [
    "data",
    "notebooks",
    "src",
    "results",
    "configs"
]

# Define the starter files
files = {
    "README.md": "# IMDb Text Classification Project\n",
    "requirements.txt": "",
    "main.py": "",
    ".gitignore": "__pycache__/\n*.pyc\n*.pyo\n.ipynb_checkpoints\ndata/\nresults/\nlogs/\n",
    "src/__init__.py": "",
    "src/preprocess.py": "",
    "src/train.py": "",
    "src/evaluate.py": "",
    "src/utils.py": "",
    "configs/default.yaml": "batch_size: 16\nepochs: 3\nlearning_rate: 2e-5\nmodel_name: bert-base-uncased\n"
}

def create_structure():
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"📂 Created folder: {folder}")

    for filepath, content in files.items():
        with open(filepath, "w") as f:
            f.write(content)
        print(f"📝 Created file: {filepath}")

if __name__ == "__main__":
    create_structure()
    print("\n✅ Project structure created successfully!")
