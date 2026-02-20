# ML Project Setup Guide
A standard, reproducible setup process for Machine Learning and Data Science projects.

---

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

---

## 2. Create Virtual Environment

Isolates project dependencies and prevents conflicts with system Python.

```bash
python -m venv .venv
```

> Same command works on Windows, Mac, and Linux.

---

## 3. Activate Virtual Environment

**Windows (PowerShell)**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt)**
```cmd
.venv\Scripts\activate.bat
```

**Mac / Linux**
```bash
source .venv/bin/activate
```

Once activated, you will see `(.venv)` at the start of your terminal line.

---

## 4. Upgrade Core Python Tools

> Note: the package is `wheel` not `wheels` — a common typo that will cause an error.

```bash
pip install --upgrade pip setuptools wheel
```

> Same command works on Windows, Mac, and Linux.

---

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
Also use backslash `\` not forward slash `/`.

```powershell
mkdir data
mkdir notebooks
mkdir src
mkdir configs
mkdir logs
mkdir tests
mkdir data\raw
mkdir data\processed
mkdir src\features
mkdir src\models
mkdir src\training
mkdir src\inference
```

---

## 6. Project Structure Overview

```
project_name/
├── data/
│   ├── raw/                        Original, untouched data. Never modify this.
│   └── processed/                  Cleaned and transformed data ready for training.
│
├── notebooks/
│   └── eda.ipynb                   Exploration, visualization, quick experiments only.
│
├── src/
│   ├── features/
│   │   └── build_features.py       Feature engineering and transformation logic.
│   ├── models/
│   │   └── model.py                Model definition or architecture.
│   ├── training/
│   │   └── train.py                Main training entry point. Run this to train.
│   └── inference/
│       └── predict.py              Loads saved model and runs predictions.
│
├── configs/
│   └── config.yaml                 All settings: paths, hyperparams, thresholds.
│
├── logs/                           Training logs and error outputs.
├── tests/
│   └── test_model.py               Unit and integration tests for src/ code.
│
├── .env                            Secret keys. Never commit this.
├── .gitignore                      Files Git should ignore.
├── requirements.txt                All packages with exact versions.
└── README.md                       Project documentation.
```

---

## 7. What Code Goes Where

**`src/features/build_features.py`**
Anything that transforms raw data before it reaches the model. Handle missing values, encode categoricals, normalize columns, create derived features.

**`src/models/model.py`**
The model definition only. Define a neural network class, load an XGBoost model, or set up a scikit-learn pipeline. No training logic lives here.

**`src/training/train.py`**
The main script you run to train. It loads data, calls feature engineering, trains the model, evaluates it, and saves it to disk.

**`src/inference/predict.py`**
Loads the saved trained model and runs predictions on new data. This is what gets called inside a FastAPI endpoint or a batch job.

**`configs/config.yaml`**
All settings that might change between runs. Nothing gets hardcoded in Python files. File paths, learning rate, epochs, model type — all go here.

**`notebooks/`**
Only for exploration. Once logic is finalized, move it into the appropriate `src/` file. Notebooks are never imported by other code.

---

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
New-Item configs\config.yaml -ItemType File
New-Item src\training\train.py -ItemType File
```

> Do not use `type nul >` in PowerShell. It does not work reliably. Use `New-Item` instead.

---

## 9. Install Common ML / DS Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
pip install mlflow fastapi uvicorn python-dotenv
```

> Same command works on Windows, Mac, and Linux.

---

## 10. Save Installed Dependencies

Freezes exact package versions so anyone can recreate the same environment.

```bash
pip freeze > requirements.txt
```

> `requirements.txt` always stays in the project root, never inside `.venv`.

---

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

> Do not use `echo x >> file` in PowerShell. It can corrupt the file encoding. Use `Add-Content` instead.

---

## 12. Write Training Entry Script

Edit `src/training/train.py`:

```python
def train():
    print("Training started")

if __name__ == "__main__":
    train()
```

---

## 13. Run Training Script

```bash
python src/training/train.py
```

**Windows (PowerShell)**
```powershell
python src\training\train.py
```

---

## 14. Initialize Git Repository

```bash
git init
git add .
git commit -m "Initial ML project setup"
```

> Same command works on Windows, Mac, and Linux.

---

## 15. Recreate Environment (If .venv Is Deleted)

The `.venv` folder is disposable and can be rebuilt anytime.

**Mac / Linux**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 16. Deactivate Virtual Environment

```bash
deactivate
```

> Same command works on Windows, Mac, and Linux.

---

## Key Rules

- Never write production code inside `.venv`
- Never commit `.venv` to GitHub
- Always activate the virtual environment before starting work
- All production code lives in `src/`
- Notebooks are for exploration only, not production
- Never hardcode settings in Python files — use `configs/config.yaml`
- Never commit `.env` to GitHub

---

> Virtual environment isolates dependencies, `requirements.txt` guarantees reproducibility, and a clean folder structure keeps ML projects maintainable and production-ready from day one.
