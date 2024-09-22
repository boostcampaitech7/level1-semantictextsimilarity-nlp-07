from src.config.path_config import BASE, OUTPUT_PATH, TRAIN_PATH
import pandas as pd
import torch
import pytorch_lightning as pl

from pathlib import Path

def save_result(data, model_name, max_epoch, filename=None, mode ='output'):
    data = list(round(float(i), 1) for i in torch.cat(data))
    if mode =='output':
        output = pd.read_csv(OUTPUT_PATH)
        output['target'] = data
        
    elif mode == 'train':
        output = pd.read_csv(TRAIN_PATH)
        output['target'] = data
    
    if filename:
        output_filename = filename
    else:
        output_filename = f'output_base_{max_epoch}_{model_name}_mse.csv'
    SAVE_PATH = Path(BASE, output_filename)
    output.to_csv(SAVE_PATH, index=False)