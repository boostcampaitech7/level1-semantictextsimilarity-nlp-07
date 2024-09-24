from src.data_loader.dataset import Dataset
from src.preprocessing.preprocessor import preprocessing, augmentation
from src.config.path_config import TRAIN_PATH,TEST_PATH,DEV_PATH,PREDICT_PATH
from src.config.data_loader_config import TRAIN_INPUT_FEATURES, TEST_INPUT_FEATURES
from torch.utils.data import DataLoader
import pandas as pd
import pytorch_lightning as pl


class Dataloader(pl.LightningDataModule):
    def __init__(self, batch_size, shuffle, model_name, check_aug=None, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

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
        self.collate_fn = None

    def setup(self, stage: str=None):
        if stage == 'fit' or stage is None:
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            if self.check_aug:
               train_data = augmentation(train_data)

            # 학습 데이터 전처리 및 준비
            train_data: pd.DataFrame = preprocessing(train_data, self.model_name)
            val_data: pd.DataFrame = preprocessing(val_data, self.model_name)

            self.train_dataset: Dataset = Dataset(train_data[TRAIN_INPUT_FEATURES])
            self.val_dataset: Dataset = Dataset(val_data[TRAIN_INPUT_FEATURES])

        if stage == 'test':
            # 테스트 데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_data: pd.DataFrame = preprocessing(test_data, self.model_name)
            self.test_dataset: Dataset = Dataset(test_data[TRAIN_INPUT_FEATURES])

            predict_data = pd.read_csv(self.predict_path)
            predict_data: pd.DataFrame = preprocessing(predict_data, self.model_name)
            self.predict_dataset: Dataset = Dataset(predict_data[TEST_INPUT_FEATURES], is_predict=True)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn
        )
