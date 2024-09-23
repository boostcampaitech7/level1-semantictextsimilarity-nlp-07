import torch
import torch.nn as nn
from tqdm.auto import tqdm
import pandas as pd
from src.config.path_config import *
import os
import glob
import numpy as np

class EnsembleModel(nn.Module):
    def __init__(self, mode,):
        super(EnsembleModel, self).__init__()
        self.mean_path = ENSEMBLE_OUTPUT
        self.linear_path = ENSEMBLE_TRAIN
        self.target_data = self.load_target_data()
        self.mode = mode
        self.num_csvs, self.mean_data = self.load_csv_data(self.mean_path, column='target')
        self.model = nn.Linear(self.num_csvs, 1).float()
        _, self.linear_data = self.load_csv_data(self.linear_path, column='label')
    
    def load_csv_data(self, path, column):
        csv_data = []
        csv_files = glob.glob(os.path.join(path, '*.csv'))
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            csv_data.append(df[column].values)
        return len(csv_files), torch.tensor(np.array(csv_data).T, dtype=torch.float32) # Shape: (num_samples, num_csvs)
    
    def load_target_data(self):
            df = pd.read_csv(TRAIN_PATH)
            return torch.tensor(df['label'].values, dtype=torch.float32).unsqueeze(1)
    
    def forward(self, data=None):
        if data is None:
            data = self.linear_data if self.mode == 'linear' else self.mean_data
        
        if self.mode == 'linear':
            return self.model(self.linear_data) 
        elif self.mode == 'mean':
            # 모든 csv의 출력 평균을 계산
            return torch.mean(self.mean_data, dim=1, keepdim=True)

def train_linear_model(model, num_epochs=100, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        outputs = model()
        loss = criterion(outputs, model.target_data)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
           
    print(f"Final loss: {loss.item()}")

def linear_inference(model, data):
    model.eval()
    with torch.no_grad():
        predictions = model(data)
    return predictions