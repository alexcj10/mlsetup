<div align="center">

# mlnew

**Professional ML project scaffolding CLI — one command, zero config.**

[![PyPI](https://img.shields.io/pypi/v/mlnew)](https://pypi.org/project/mlnew/)

</div>

## Quick Start

```bash
pip install mlnew
```

```bash
mlnew init my_project
```

That's it. Your entire ML workspace is ready.

> Run `pip install -U mlnew` to always get the latest version.

## What It Does

A single `mlnew init` command will:

| Step | Action |
|:--:|:--|
| 1 | Create a professional folder structure (`data`, `src`, `notebooks`, `configs`, `logs`, `tests`) |
| 2 | Generate essential files (`.gitignore`, `.env`, `config.yaml`, `train.py`, `eda.ipynb`) |
| 3 | Include a full `SETUP_GUIDE.md` manual inside every project |
| 4 | Set up a virtual environment (`.venv`) with `pip`, `setuptools`, `wheel` |
| 5 | Install 10 core ML/DS packages (NumPy, Pandas, Scikit-learn, etc.) |
| 6 | Pin exact versions in `requirements.txt` |
| 7 | Generate a project-specific `README.md` |
| 8 | Initialize Git with an initial commit |

## Default Packages

All packages install the **latest compatible version** automatically.

| Package | Description |
|:--|:--|
| `numpy` | Numerical computing |
| `pandas` | Data manipulation |
| `scikit-learn` | Machine learning algorithms |
| `matplotlib` | Visualization |
| `seaborn` | Statistical visualization |
| `jupyter` | Interactive notebooks |
| `mlflow` | Experiment tracking |
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server |
| `python-dotenv` | Environment variable management |

```bash
mlnew packages     # View all defaults anytime
```

> Note: These 10 core packages include their own transitive dependencies (~150+ total), all of which are auto-detected and pinned in `requirements.txt`.

## Customize Packages

Override any default or add new packages with `--pkg`:

```bash
# Pin specific versions
mlnew init my_project --pkg numpy==1.24.0 --pkg pandas==2.0.0

# Install latest (no pin)
mlnew init my_project --pkg numpy==latest

# Add packages not in defaults
mlnew init my_project --pkg torch --pkg transformers

# Mix and match
mlnew init my_project --pkg numpy==1.24.0 --pkg torch --pkg transformers==4.40.0
```

## Project Structure

```
my_project/
├── .venv/                 Virtual environment (auto-created)
├── data/
│   ├── raw/               Original, untouched data
│   └── processed/         Cleaned and transformed data
├── notebooks/
│   └── eda.ipynb          Exploration and visualization
├── src/
│   ├── features/          Feature engineering
│   ├── models/            Model definitions
│   ├── training/
│   │   └── train.py       Training entry point
│   └── inference/         Prediction and serving
├── configs/
│   └── config.yaml        Settings and hyperparameters
├── logs/                  Training logs
├── tests/                 Unit and integration tests
├── .env                   Secrets (never committed)
├── .gitignore
├── requirements.txt       Pinned dependencies
├── SETUP_GUIDE.md         Full manual reference
└── README.md
```

## After Setup

```bash
cd my_project

# Activate virtual environment
source .venv/bin/activate          # Mac / Linux
.venv\Scripts\Activate.ps1         # Windows (PowerShell)

# Start training
python src/training/train.py
```

## All Commands

| Command | Description |
|:--|:--|
| `mlnew init <name>` | Create project with default packages |
| `mlnew init <name> --pkg <spec>` | Override specific packages |
| `mlnew packages` | List default packages and versions |
| `mlnew --version` | Show version |
| `mlnew --help` | Show help |

## Requirements

- **Python** 3.8+
- **Git** (optional, for auto `git init`)

## Troubleshooting

<details>
<summary><b>Command not found: <code>mlnew</code> (Windows)</b></summary>

If you see `mlnew: The term 'mlnew' is not recognized...`, your Python Scripts folder is not in PATH.

**Fix PATH (Recommended):**
1. Search Windows for *"Edit the system environment variables"*
2. Click *"Environment Variables"*
3. Under *"User variables"*, find `Path` and click *"Edit"*
4. Add your Python Scripts folder (e.g., `C:\Users\YourName\AppData\Roaming\Python\Python313\Scripts`)
5. Restart your terminal

**Or use Python module directly:**
```bash
python -m mlnew init my_project
```

</details>

## License

[MIT](LICENSE)
