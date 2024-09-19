import pytorch_lightning as pl
import transformers
import torch
import torchmetrics
from dataclasses import dataclass, asdict
from typing import Callable
from utils.similarity import *


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

    def __init__(self, model_name):
        self._model_name = model_name
        assert self._model_name in asdict(self).values()
    
    @property
    def plm(self):
        return transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self._model_name, num_labels=1)

    def get_model_name(self, name):
        if not hasattr(self, name):
            raise ValueError(f"'{name}' is not a valid model name in Models class \n select Model -> {', '.join(list(asdict(self).keys()))}")
        return getattr(self, name)


class PlmObject:
    def __init__(self, model_name):
        self.model = Models(model_name).plm


class Model(pl.LightningModule):
    def __init__(self, model_name, lr, loss_func):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        # 사용할 모델을 호출합니다.
        self.plm = PlmObject(model_name).model
        assert self.plm is not None

        # Loss 계산을 위해 사용할 Loss func 사용
        self.loss_func = loss_func
        assert self.loss_func is not None

        # 유사도 피처를 처리할 추가 레이어
        self.similarity_layer = torch.nn.Linear(3, 1)
        
    def forward(self, x, sentence1, sentence2, attention_mask=None):
        # 문장간 유사도 계산 (lsa, jaccard, fuzzy)
        lsa = lsa_similarity(sentence1, sentence2)
        jaccard = jaccard_similarity(sentence1, sentence2)
        fuzzy = fuzzy_similarity(sentence1, sentence2)

        # 유사도 정규화
        lsa, jaccard, fuzzy = normalize_similarity(lsa, jaccard, fuzzy)

        # 유사도 피처를 하나로 합침
        similarity_features = torch.tensor([lsa, jaccard, fuzzy], device=x.device).unsqueeze(0).float()
        
        # PLM을 통해 원래의 예측값 계산
        logits = self.plm(input_ids=x, attention_mask=attention_mask)['logits']
        
        # 유사도 피처를 추가 처리하고 최종 출력에 합침
        similarity_logits = self.similarity_layer(similarity_features)
        logits += similarity_logits

        return logits

    def training_step(self, batch, batch_idx):
        # 배치 데이터를 리스트의 인덱스로 접근
        x, y, sentence1, sentence2 = batch[0], batch[1], batch[2], batch[3]
        logits = self(x, sentence1, sentence2).squeeze()
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        # 배치 데이터를 리스트의 인덱스로 접근
        x, y, sentence1, sentence2 = batch[0], batch[1], batch[2], batch[3]
        logits = self(x, sentence1, sentence2)
        loss = self.loss_func(logits, y.float())
        val_pearson = torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze())
        self.log("val_loss", loss)
        self.log("val_pearson", val_pearson)

        return loss

    def test_step(self, batch, batch_idx):
        # 배치 데이터를 리스트의 인덱스로 접근
        x, y, sentence1, sentence2 = batch[0], batch[1], batch[2], batch[3]
        logits = self(x, sentence1, sentence2)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        # 배치 데이터를 리스트의 인덱스로 접근
        x, sentence1, sentence2 = batch[0], batch[1], batch[2]
        logits = self(x, sentence1, sentence2)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
