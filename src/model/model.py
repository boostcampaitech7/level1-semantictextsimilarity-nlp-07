import pytorch_lightning as pl
import transformers
import torch
import torchmetrics
from dataclasses import dataclass, asdict
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
    electra_base: str = "snunlp/KR-ELECTRA-discriminator"
    electra_base_v3: str = "monologg/koelectra-base-v3-discriminator"
    roberta_base: str = "klue/roberta-base"
    roberta_small: str = "klue/roberta-small"
    roberta_large: str = "klue/roberta-large"
    t5_base: str = "sentence-transformers/sentence-t5-base"
    t5_gtr_base: str = "sentence-transformers/gtr-t5-base"
    sroberta: str = "jhgan/ko-sroberta-multitask"
    roberta_base_trending: str = "bespin-global/klue-sroberta-base-continue-learning-by-mnr"

    def __init__(self, model_name, num_labels:int = 1):
        self._model_name = model_name
        self.num_labels = num_labels
        assert self._model_name in asdict(self).values()
    
    @property
    def plm(self):
        return transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self._model_name, num_labels=self.num_labels)

    def get_model_name(self, name):
        if not hasattr(self, name):
            raise ValueError(f"'{name}' is not a valid model name in Models class \n select Model -> {', '.join(list(asdict(self).keys()))}")
        return getattr(self, name)


class PlmObject:
    def __init__(self, model_name, num_labels):
        self.model = Models(model_name, num_labels).plm


class Model(pl.LightningModule):
    def __init__(self, model_name: str, lr: int, loss_func: Callable, mode: str ='regression'):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.lr = lr
        
        self.mode = mode
        if mode == 'regression':
            self.num_labels = 1
            self.loss_func = loss_func
        elif mode == 'classification':
            self.num_labels = 31
            self.loss_func = torch.nn.CrossEntropyLoss()
        self.class_values = torch.tensor(CLASSIFICATION_VALUES)
            
        # 사용할 모델을 호출합니다.
        self.plm = PlmObject(model_name, self.num_labels).model
        assert self.plm is not None
        
    def forward(self, input_data, attention_mask):
        logits = self.plm(input_ids=input_data, attention_mask=attention_mask)['logits']
        return logits.squeeze()

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

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        input_ids = batch[INPUT_IDS_INDEX]
        attention_mask = batch[ATTENTION_MASK_INDEX]
        
        outputs = self(input_ids, attention_mask)
        if self.mode == 'classification':
            pred_class = torch.argmax(outputs, dim=1)
            predictions = self.class_values[pred_class]
        else:  # regression mode
            predictions = outputs
        
        # dataloader_idx에 따라 다른 처리를 할 수 있습니다
        if dataloader_idx == 0:
            return {"test_predictions": predictions}
        else:
            return {"val_predictions": predictions}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
    
    def _get_class_index(self, label):
        return torch.argmin(torch.abs(self.class_values.unsqueeze(0) - label.unsqueeze(1)), dim=1)
