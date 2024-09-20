import torch
import pandas as pd


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame, is_predict=False):
        self.data = data
        self.is_predict = is_predict

    def __getitem__(self, idx: int):
        if self.is_predict:
            input_ids: torch.Tensor = torch.tensor(self.data.iloc[idx]['input_ids'], dtype=torch.long)
            attention_mask: torch.Tensor = torch.tensor(self.data.iloc[idx]['attention_mask'], dtype=torch.long)
            return input_ids, attention_mask

        input_ids: torch.Tensor = torch.tensor(self.data.iloc[idx]['input_ids'], dtype=torch.long)
        attention_mask: torch.Tensor = torch.tensor(self.data.iloc[idx]['attention_mask'], dtype=torch.long)
        label: torch.Tensor = torch.tensor(self.data.iloc[idx]['label'], dtype=torch.float32)
        return input_ids, attention_mask, label
    
    def __len__(self):
        return len(self.data)
