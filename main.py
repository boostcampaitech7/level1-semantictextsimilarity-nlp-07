from src.config.data_loader_config import DATA_LOADER_CONFIG, OPTIMIZER_CONFIG
from src.data_loader.loader import Dataloader
from src.model.model import Model, Models, LossFunctions
from src.trainer.predict import save_result
import src.callback as callback
import pytorch_lightning as pl
import torch
import os

# Parameters 선언
batch_size: int = DATA_LOADER_CONFIG['batch_size']
shuffle: bool = DATA_LOADER_CONFIG['shuffle']
learning_rate: float = OPTIMIZER_CONFIG['learning_rate']
max_epoch: int = OPTIMIZER_CONFIG['max_epoch']
num_workers: int = DATA_LOADER_CONFIG.get('num_workers', 4)  # num_workers 기본값 4로 설정

if __name__ == "__main__":
    torch.cuda.empty_cache()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model_name = 'electra_base_v3'
    model = Model(Models.electra_base_v3, learning_rate, LossFunctions.mse_loss)
    print('Calling Model is Successful')
    
    # Dataloader 선언
    dataloader = Dataloader(batch_size, shuffle, model_name=Models.electra_base_v3, num_workers=num_workers)
    print('Calling Dataloader')
    
    # callback 정의
    epoch_print_callback = callback.EpochPrintCallback()
    checkpoint_callback = callback.ModelCheckpoint(model_name=model_name)
    early_stopping = callback.EarlyStopping()
    lr_monitor = callback.LearningRateMonitor()
    print('Calling Callback')
    
    # 학습 및 검증
    print('Start Training')
    trainer = pl.Trainer(
        accelerator='auto',
        devices='auto',
        max_epochs=max_epoch,
        callbacks=[lr_monitor, epoch_print_callback, checkpoint_callback, early_stopping],
        precision='16-mixed'
    )
    trainer.fit(model=model, datamodule=dataloader)
    
    print('------'*20)
    trainer.test(model=model, datamodule=dataloader)
    
    # 가장 좋은 모델 불러오기
    best_model_path = checkpoint_callback.best_model_path
    model = Model.load_from_checkpoint(
        best_model_path, 
        model_name=Models.electra_base_v3,
        lr=learning_rate,
        loss_func=LossFunctions.mse_loss
    )
    print('----------'*10, 'Start Prediction', '----------'*10)
    
    # 추론
    predictions = trainer.predict(model=model, datamodule=dataloader)

    # 결과 저장
    save_result(predictions, model_name, max_epoch)
    print('----------'*10, 'Finish', '----------'*10)
