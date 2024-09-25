from src.config.path_config import BASE, OUTPUT_PATH, DEV_PATH
import pandas as pd
import torch
import pytorch_lightning as pl
from pathlib import Path

def save_result(predictions, model_name, max_epoch, mode='output'):
    key = 'test_predictions' if mode == 'output' else 'val_predictions'
    all_predictions = torch.cat([pred[key] for pred in predictions])
    
    data = [round(float(i), 1) for i in all_predictions.flatten()]
    
    if mode == 'output':
        output = pd.read_csv(OUTPUT_PATH)
        output['target'] = data
        output_filename = f'output/output_{max_epoch}_{model_name}.csv'
    elif mode == 'train':
        output = pd.read_csv(DEV_PATH)
        output['target'] = data
        output_filename = f'train/train_{max_epoch}_{model_name}.csv'

    SAVE_PATH = Path(BASE,'results', output_filename)
    output.to_csv(SAVE_PATH, index=False)