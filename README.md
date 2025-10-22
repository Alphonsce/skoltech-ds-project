# Skoltech DS Project, 2025 Fall
- [Competition](https://zindi.africa/competitions/expresso-churn-prediction)

## Installation:

```bash
pip install uv

uv venv --python=3.10
uv pip install -r requirements.txt
source .venv/bin/activate
```

## Project Structure:

- EDA is done inside `StarterNotebook.ipynb`

- Main notebook with all the logic is `main.ipynb`

- Code for NaN imputation is done in `src/imputers.py`

- Code for model training, optimization and evaluation is in `src/catboost.py`

- Utility functions are in `src/utils.py`