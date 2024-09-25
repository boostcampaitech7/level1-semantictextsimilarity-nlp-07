import pytorch_lightning as pl
import transformers
import torch
import torchmetrics
from dataclasses import dataclass
from typing import Callable
# from utils.similarity import *
from src.config.data_loader_config import *
import pandas as pd


INPUT_IDS_INDEX = TRAIN_INPUT_FEATURES.index('input_ids')
ATTENTION_MASK_INDEX = TRAIN_INPUT_FEATURES.index('attention_mask')
LABEL_INDEX = TRAIN_INPUT_FEATURES.index('label')


@dataclass
class LossFunctions:
    mse_loss: Callable = torch.nn.MSELoss()  # object id 값만 저장
    l1_loss: Callable = torch.nn.L1Loss()
    hu_loss: Callable = torch.nn.HuberLoss()
    
@dataclass
class Models:
    model_dict = {'electra_base': "snunlp/KR-ELECTRA-discriminator",
    'electra_base_v3': "monologg/koelectra-base-v3-discriminator",
    'roberta_base': "klue/roberta-base",
    'roberta_small': "klue/roberta-small",
    'roberta_large': "klue/roberta-large",
    'synatra7b': "maywell/Synatra-7B-v0.3-dpo",
    'saltlux': "saltlux/Ko-Llama3-Luxia-8B",
    'ko-soberta': "jhgan/ko-sroberta-multitask"}

    def __init__(self, model_name):
        self._model_name = model_name
    
    @property
    def plm(self):
        return transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self._model_name, torch_dtype='auto', num_labels=1)

    def get_model_name(self, name):
        if not hasattr(self, name):
            raise ValueError(f"'{name}' is not a valid model name in Models class \n select Model -> {', '.join(list(self.model_dict.keys()))}")
        return getattr(self, name)


class PlmObject:
    def __init__(self, model_name):
        self.model = Models(model_name).plm


class Model(pl.LightningModule):
    def __init__(self, model_name: str, lr: int, loss_func: Callable):
        super().__init__()
        self.save_hyperparameters(ignore=['loss_func'])

        self.model_name = model_name
        self.lr = lr

        # 사용할 모델을 호출합니다.
        self.plm = PlmObject(model_name).model
        assert self.plm is not None

        # Loss 계산을 위해 사용할 Loss func 사용
        self.loss_func = loss_func
        assert self.loss_func is not None
        
    def forward(self, input_data, attention_mask):
        logits = self.plm(input_ids=input_data, attention_mask=attention_mask)['logits']
        return logits

    def training_step(self, batch, batch_idx):
        input_ids = batch[INPUT_IDS_INDEX]
        attention_mask = batch[ATTENTION_MASK_INDEX]
        label = batch[LABEL_INDEX]
        
        outputs: torch.Tensor = self(input_ids, attention_mask)
        loss: torch.Tensor = self.loss_func(outputs, label)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch[INPUT_IDS_INDEX]
        attention_mask = batch[ATTENTION_MASK_INDEX]
        label = batch[LABEL_INDEX]
        
        outputs: torch.Tensor = self(input_ids, attention_mask)
        loss: torch.Tensor = self.loss_func(outputs, label)
        val_pearson: torch.Tensor = torchmetrics.functional.pearson_corrcoef(outputs.squeeze(), label.squeeze())
        self.log("val_loss", loss)
        self.log("val_pearson", val_pearson)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch[INPUT_IDS_INDEX]
        attention_mask = batch[ATTENTION_MASK_INDEX]
        label = batch[LABEL_INDEX]
        
        outputs: torch.Tensor = self(input_ids, attention_mask)
        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(outputs.squeeze(), label.squeeze()))

    def predict_step(self, batch, batch_idx):
        input_ids = batch[INPUT_IDS_INDEX]
        attention_mask = batch[ATTENTION_MASK_INDEX]
        
        outputs: torch.Tensor = self(input_ids, attention_mask)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
