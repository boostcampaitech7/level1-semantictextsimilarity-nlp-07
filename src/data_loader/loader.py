from src.data_loader.dataset import Dataset
from src.preprocessing.preprocessor import preprocessing, augmentation
from src.config.path_config import TRAIN_PATH,TEST_PATH,DEV_PATH,PREDICT_PATH
import torch
import pandas as pd
import pytorch_lightning as pl


class Dataloader(pl.LightningDataModule):
    def __init__(self, batch_size, shuffle, model_name, check_aug=None):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = TRAIN_PATH
        self.dev_path = DEV_PATH
        self.test_path = TEST_PATH
        self.predict_path = PREDICT_PATH
        self.model_name = model_name
        self.check_aug = check_aug
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)
            if self.check_aug == True:
               train_data = augmentation(train_data)
            # 학습데이터 준비
            train_inputs, train_targets = preprocessing(train_data, self.model_name)

            # 검증데이터 준비
            val_inputs, val_targets = preprocessing(val_data, self.model_name)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = preprocessing(test_data, self.model_name)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = preprocessing(predict_data, self.model_name)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)