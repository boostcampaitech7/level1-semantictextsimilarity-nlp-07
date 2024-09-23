from pathlib import Path 
from src.config.data_loader_config import DATA_LOADER_CONFIG


BASE = Path(__file__).resolve().parent.parent.parent

TRAIN_PATH = Path(BASE, DATA_LOADER_CONFIG['train_path'])
DEV_PATH = Path(BASE, DATA_LOADER_CONFIG['dev_path'])
TEST_PATH = Path(BASE, DATA_LOADER_CONFIG['test_path'])
PREDICT_PATH = Path(BASE, DATA_LOADER_CONFIG['predict_path'])
OUTPUT_PATH = Path(BASE, DATA_LOADER_CONFIG['output_path'])

ENSEMBLE_TRAIN = Path(BASE, 'results', 'train')
ENSEMBLE_OUTPUT = Path(BASE, 'results', 'output')
