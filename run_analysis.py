from pathlib import Path
import os
import sys

os.environ.setdefault('MPLBACKEND', 'Agg')

PROJECT_ROOT = Path(__file__).resolve().parents[0]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.pipeline import run_pipeline
from src.config import FIGURES_DIR, TABLES_DIR

if __name__ == '__main__':
    results = run_pipeline()
    print('Analysis complete.')
    print(f"Saved figures to: {FIGURES_DIR}")
    print(f"Saved tables to: {TABLES_DIR}")
    if not results['best_models'].empty:
        print('\nBest model by country:')
        print(results['best_models'][['country', 'model', 'RMSE']].to_string(index=False))
