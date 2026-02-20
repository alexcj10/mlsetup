import sys
import subprocess
import platform
import venv
from pathlib import Path


# Default packages with their latest stable versions
DEFAULT_PACKAGES = {
    "numpy":         None,
    "pandas":        None,
    "scikit-learn":  None,
    "matplotlib":    None,
    "seaborn":       None,
    "jupyter":       None,
    "mlflow":        None,
    "fastapi":       None,
    "uvicorn":       None,
    "python-dotenv": None,
}

DIRECTORIES = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src/features",
    "src/models",
    "src/training",
    "src/inference",
    "configs",
    "logs",
    "tests",
]

FILES = {
    "requirements.txt": "",
    ".env": "",
    "configs/config.yaml": "# Project configuration\n",
    "src/__init__.py": "",
    "src/features/__init__.py": "",
    "src/models/__init__.py": "",
    "src/training/__init__.py": "",
    "src/inference/__init__.py": "",
    "tests/__init__.py": "",
    ".gitignore": ".venv/\n__pycache__/\n.ipynb_checkpoints/\n.env\n*.log\n*.pyc\ndist/\nbuild/\n*.egg-info/\n",
    "src/training/train.py": (
        "def train():\n"
        "    print(\"Training started\")\n\n"
        "if __name__ == \"__main__\":\n"
        "    train()\n"
    ),
    "notebooks/eda.ipynb": (
        '{\n "cells": [],\n "metadata": {\n  "kernelspec": {\n'
        '   "display_name": "Python 3",\n   "language": "python",\n'
        '   "name": "python3"\n  }\n },\n "nbformat": 4,\n "nbformat_minor": 5\n}\n'
    ),
}

IS_WINDOWS = platform.system() == "Windows"


def green(text):  return f"\033[92m{text}\033[0m"
def yellow(text): return f"\033[93m{text}\033[0m"
def red(text):    return f"\033[91m{text}\033[0m"
def bold(text):   return f"\033[1m{text}\033[0m"
def dim(text):    return f"\033[2m{text}\033[0m"
def cyan(text):   return f"\033[96m{text}\033[0m"


def step(msg):  print(f"  {green('✓')} {msg}")
def info(msg):  print(f"  {dim('→')} {msg}")
def warn(msg):  print(f"  {yellow('⚠')} {msg}")
def error(msg):
    print(f"  {red('✗')} {msg}")
    sys.exit(1)


def get_pip_path(project_path: Path) -> str:
    if IS_WINDOWS:
        return str(project_path / ".venv" / "Scripts" / "pip.exe")
    return str(project_path / ".venv" / "bin" / "pip")


def parse_packages(args: list) -> dict:
    """
    Parse --pkg flags from args.
    Supports:
      --pkg numpy              → install latest (default version)
      --pkg numpy==1.24.0     → install exact version
      --pkg numpy==latest     → install latest, no pin
    Returns dict of {package_name: version_spec or None}
    """
    packages = {}
    i = 0
    while i < len(args):
        if args[i] == "--pkg" and i + 1 < len(args):
            spec = args[i + 1]
            if "==" in spec:
                name, version = spec.split("==", 1)
                packages[name.strip()] = None if version.strip() == "latest" else version.strip()
            else:
                name = spec.strip()
                packages[name] = DEFAULT_PACKAGES.get(name, None)
            i += 2
        else:
            i += 1
    return packages


def build_install_list(packages: dict) -> list:
    """Turn {name: version} dict into pip install strings."""
    result = []
    for name, version in packages.items():
        if version:
            result.append(f"{name}=={version}")
        else:
            result.append(name)
    return result


def show_packages_list():
    print()
    print(bold("  Default packages:"))
    print()
    for pkg, ver in DEFAULT_PACKAGES.items():
        display_ver = ver if ver else "latest"
        print(f"    {cyan(pkg):<30} {dim(display_ver)}")
    print()
    print("  Override any package version with --pkg:")
    print(f"    {dim('mlnew init myproject --pkg numpy==1.24.0 --pkg pandas==2.0.0')}")
    print(f"    {dim('mlnew init myproject --pkg numpy==latest')}")
    print(f"    {dim('mlnew init myproject --pkg torch --pkg transformers')}")
    print()


def create_project(project_name: str, packages: dict):
    project_path = Path.cwd() / project_name

    print()
    print(bold(f"  Setting up ML project: {project_name}"))
    print()

    # 1. Create root folder
    if project_path.exists():
        error(f"Folder '{project_name}' already exists in this directory.")
    project_path.mkdir()
    step(f"Created project folder  →  {project_path}")

    # 2. Create directories
    for d in DIRECTORIES:
        (project_path / d).mkdir(parents=True, exist_ok=True)
    step("Created folder structure  →  data, src, notebooks, configs, logs, tests")

    # 3. Create files
    for filepath, content in FILES.items():
        full_path = project_path / filepath
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
    step("Created required files  →  .gitignore, .env, config.yaml, train.py, eda.ipynb")

    # 3b. Write README (Early creation to ensure existence)
    (project_path / "README.md").write_text(generate_readme(project_name, packages), encoding="utf-8")
    step("Generated README.md")

    # 4. Write SETUP_GUIDE.md (the full manual reference)
    (project_path / "SETUP_GUIDE.md").write_text(generate_setup_guide(), encoding="utf-8")
    step("Generated SETUP_GUIDE.md  →  full manual reference included")

    # 5. Create virtual environment
    info("Creating virtual environment (.venv) ...")
    venv.create(str(project_path / ".venv"), with_pip=True)
    step("Created virtual environment  →  .venv/")

    # 6. Upgrade pip
    pip = get_pip_path(project_path)
    info("Upgrading pip, setuptools, wheel ...")
    subprocess.run(
        [pip, "install", "--upgrade", "pip", "setuptools", "wheel"],
        capture_output=True
    )
    step("Upgraded pip, setuptools, wheel")

    # 7. Install dependencies
    install_list = build_install_list(packages)
    info(f"Installing {len(install_list)} packages ...")
    for pkg in install_list:
        info(f"  pip install {pkg}")

    result = subprocess.run(
        [pip, "install"] + install_list,
        check=False  # Allow getting return code manually
    )
    if result.returncode != 0:
        warn("Some packages failed to install. Check the output above.")
        # warn(result.stderr.strip().splitlines()[-1] if result.stderr else "Unknown error")
    else:
        step(f"Installed {len(install_list)} packages successfully")

    # 8. Freeze requirements
    freeze_result = subprocess.run([pip, "freeze"], capture_output=True, text=True)
    if not freeze_result.stdout.strip():
        warn("requirements.txt is empty! Dependencies might not have installed correctly.")
    (project_path / "requirements.txt").write_text(freeze_result.stdout, encoding="utf-8")
    step("Saved requirements.txt  →  exact versions pinned")

    # 9. Write README (Moved to step 3b)
    # (project_path / "README.md").write_text(generate_readme(project_name, packages), encoding="utf-8")
    # step("Generated README.md")

    # 10. Git init
    git_result = subprocess.run(
        ["git", "init", str(project_path)],
        capture_output=True, text=True
    )
    if git_result.returncode == 0:
        subprocess.run(["git", "-C", str(project_path), "add", "."], capture_output=True)
        subprocess.run(
            ["git", "-C", str(project_path), "commit", "-m", "Initial ML project setup"],
            capture_output=True
        )
        step("Initialized Git repository  →  initial commit done")
    else:
        warn("Git not found. Skipping git init.")

    # Done
    activate_cmd = (
        r".venv\Scripts\Activate.ps1" if IS_WINDOWS
        else "source .venv/bin/activate"
    )
    print()
    print(bold(f"  Project ready at ./{project_name}/"))
    print()
    print("  Next steps:")
    print(f"    cd {project_name}")
    print(f"    {activate_cmd}")
    print(f"    python src/training/train.py")
    print()
    print(f"  {dim('Full manual reference → SETUP_GUIDE.md')}")
    print()


def generate_readme(project_name: str, packages: dict) -> str:
    pkg_lines = "\n".join(
        f"- `{name}=={ver}`" if ver else f"- `{name}` (latest)"
        for name, ver in packages.items()
    )
    activate = r".venv\Scripts\Activate.ps1" if IS_WINDOWS else "source .venv/bin/activate"
    return f"""# {project_name}

ML project scaffolded with [mlnew](https://pypi.org/project/mlnew/).

## Setup

```bash
python -m venv .venv
{activate}
pip install -r requirements.txt
```

## Run Training

```bash
python src/training/train.py
```

## Top-Level Packages

{pkg_lines}

> Note: These core packages include their own dependencies (approx. 150+ total), which are documented and pinned in `requirements.txt`.

## Project Structure

```
{project_name}/
├── data/
│   ├── raw/               Original, untouched data.
│   └── processed/         Cleaned and transformed data.
├── notebooks/             EDA and exploration only.
├── src/
│   ├── features/          Feature engineering logic.
│   ├── models/            Model definition.
│   ├── training/          Training entry point.
│   └── inference/         Prediction and serving logic.
├── configs/               YAML config files.
├── logs/                  Training logs.
├── tests/                 Unit and integration tests.
├── .env                   Secret keys. Never commit.
├── .gitignore
├── requirements.txt
├── SETUP_GUIDE.md         Full manual setup reference.
└── README.md
```

## Key Rules

- Never write production code inside `.venv`
- Never commit `.venv` or `.env` to GitHub
- All production code lives in `src/`
- Notebooks are for exploration only
- All settings go in `configs/config.yaml`, never hardcoded
"""


def generate_setup_guide() -> str:
    guide_path = __import__('pathlib').Path(__file__).parent.parent / "SETUP_GUIDE.md"
    if guide_path.exists():
        return guide_path.read_text(encoding="utf-8")
    return """\
# ML Project Setup Guide

A standard, reproducible setup process for Machine Learning and Data Science projects.

## 1. Create Project Folder

```bash
# Mac / Linux
mkdir project_name
cd project_name
```

```powershell
# Windows (PowerShell)
mkdir project_name
cd project_name
```

## 2. Create Virtual Environment

Isolates project dependencies and prevents conflicts with system Python.

```bash
python -m venv .venv
```

> Same command works on Windows, Mac, and Linux.

## 3. Activate Virtual Environment

| Platform | Command |
|:--|:--|
| Windows (PowerShell) | `.venv\\Scripts\\Activate.ps1` |
| Windows (CMD) | `.venv\\Scripts\\activate.bat` |
| Mac / Linux | `source .venv/bin/activate` |

Once activated, you will see `(.venv)` at the start of your terminal line.

## 4. Upgrade Core Python Tools

> Note: the package is `wheel` not `wheels` — a common typo that will cause an error.

```bash
pip install --upgrade pip setuptools wheel
```

## 5. Create Project Folder Structure

**Mac / Linux**
```bash
mkdir -p data/raw data/processed
mkdir -p notebooks
mkdir -p src/features src/models src/training src/inference
mkdir -p configs logs tests
```

**Windows (PowerShell)**

PowerShell only accepts one folder per `mkdir` command. Run each line one by one.
Also use backslash `\\` not forward slash `/`.

```powershell
mkdir data
mkdir notebooks
mkdir src
mkdir configs
mkdir logs
mkdir tests
mkdir data\\raw
mkdir data\\processed
mkdir src\\features
mkdir src\\models
mkdir src\\training
mkdir src\\inference
```

## 6. Project Structure Overview

```
project_name/
├── data/
│   ├── raw/                        Original, untouched data
│   └── processed/                  Cleaned and transformed data
├── notebooks/
│   └── eda.ipynb                   Exploration and visualization only
├── src/
│   ├── features/
│   │   └── build_features.py       Feature engineering logic
│   ├── models/
│   │   └── model.py                Model definition or architecture
│   ├── training/
│   │   └── train.py                Main training entry point
│   └── inference/
│       └── predict.py              Prediction and serving logic
├── configs/
│   └── config.yaml                 Settings, paths, hyperparameters
├── logs/                           Training logs and error outputs
├── tests/
│   └── test_model.py               Unit and integration tests
├── .env                            Secret keys (never commit)
├── .gitignore                      Files Git should ignore
├── requirements.txt                All packages with exact versions
└── README.md                       Project documentation
```

## 7. What Code Goes Where

| File | Purpose |
|:--|:--|
| `src/features/build_features.py` | Transforms raw data. Handle missing values, encode categoricals, normalize columns, create derived features. |
| `src/models/model.py` | Model definition only. No training logic lives here. |
| `src/training/train.py` | Main script to train. Loads data, trains model, evaluates, and saves to disk. |
| `src/inference/predict.py` | Loads saved model and runs predictions. Called inside a FastAPI endpoint or batch job. |
| `configs/config.yaml` | All settings that might change between runs. Never hardcoded in Python. |
| `notebooks/` | Exploration only. Move finalized logic into `src/`. Never imported by other code. |

## 8. Create Required Files

**Mac / Linux**
```bash
touch requirements.txt .gitignore README.md .env
touch configs/config.yaml
touch src/training/train.py
```

**Windows (PowerShell)**
```powershell
New-Item requirements.txt, .gitignore, README.md, .env -ItemType File
New-Item configs\\config.yaml -ItemType File
New-Item src\\training\\train.py -ItemType File
```

> Do not use `type nul >` in PowerShell. Use `New-Item` instead.

## 9. Install Common ML / DS Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
pip install mlflow fastapi uvicorn python-dotenv
```

> Same command works on Windows, Mac, and Linux.

## 10. Save Installed Dependencies

```bash
pip freeze > requirements.txt
```

> `requirements.txt` always stays in the project root, never inside `.venv`.

## 11. Setup .gitignore

**Mac / Linux**
```bash
echo ".venv/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo ".ipynb_checkpoints/" >> .gitignore
echo ".env" >> .gitignore
echo "*.log" >> .gitignore
```

**Windows (PowerShell)**
```powershell
Add-Content .gitignore ".venv/"
Add-Content .gitignore "__pycache__/"
Add-Content .gitignore ".ipynb_checkpoints/"
Add-Content .gitignore ".env"
Add-Content .gitignore "*.log"
```

> Do not use `echo x >> file` in PowerShell. Use `Add-Content` instead.

## 12. Write Training Entry Script

Edit `src/training/train.py`:

```python
def train():
    print("Training started")

if __name__ == "__main__":
    train()
```

## 13. Run Training Script

```bash
python src/training/train.py
```

**Windows (PowerShell)**
```powershell
python src\\training\\train.py
```

## 14. Initialize Git Repository

```bash
git init
git add .
git commit -m "Initial ML project setup"
```

> Same command works on Windows, Mac, and Linux.

## 15. Recreate Environment (If .venv Is Deleted)

**Mac / Linux**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

## 16. Deactivate Virtual Environment

```bash
deactivate
```

> Same command works on Windows, Mac, and Linux.

## Key Rules

| Rule |
|:--|
| Never write production code inside `.venv` |
| Never commit `.venv` to GitHub |
| Always activate the virtual environment before starting work |
| All production code lives in `src/` |
| Notebooks are for exploration only, not production |
| Never hardcode settings in Python files — use `configs/config.yaml` |
| Never commit `.env` to GitHub |

> Virtual environment isolates dependencies, `requirements.txt` guarantees reproducibility, and a clean folder structure keeps ML projects maintainable and production-ready from day one.
"""


def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print()
        print(bold("  mlnew — ML Project Scaffolding CLI"))
        print()
        print("  Usage:")
        print(f"    {cyan('mlnew init <project_name>')}              Create project with default packages")
        print(f"    {cyan('mlnew init <project_name> --pkg <spec>')} Override specific package versions")
        print(f"    {cyan('mlnew packages')}                         List default packages and versions")
        print(f"    {cyan('mlnew --version')}                        Show version")
        print()
        print("  Examples:")
        print(f"    {dim('mlnew init my_project')}")
        print(f"    {dim('mlnew init my_project --pkg numpy==1.24.0 --pkg pandas==2.0.0')}")
        print(f"    {dim('mlnew init my_project --pkg numpy==latest --pkg torch')}")
        print()
        return

    if args[0] == "--version":
        try:
            from importlib.metadata import version as pkg_version
            ver = pkg_version("mlnew")
        except Exception:
            ver = "unknown"
        print(f"  mlnew version {ver}")
        return

    if args[0] == "packages":
        show_packages_list()
        return

    if args[0] == "init":
        if len(args) < 2:
            error("Please provide a project name.  Usage: mlnew init <project_name>")
        project_name = args[1]
        if not project_name.replace("_", "").replace("-", "").isalnum():
            error("Project name can only contain letters, numbers, hyphens, and underscores.")

        # Parse any --pkg overrides
        overrides = parse_packages(args[2:])

        # Start with defaults, apply overrides on top
        final_packages = dict(DEFAULT_PACKAGES)
        for name, version in overrides.items():
            final_packages[name] = version  # override version or add new package

        create_project(project_name, final_packages)
        return

    error(f"Unknown command: '{args[0]}'. Run 'mlnew --help' for usage.")


if __name__ == "__main__":
    main()
