from pytorch_lightning.callbacks import Callback
class EpochPrintCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Epoch {trainer.current_epoch} ended")
        metrics = trainer.callback_metrics
        # 검증 손실 출력 (만약 검증을 수행했다면)
        if trainer.callback_metrics.get("val_loss"):
            print(f"Validation Loss: {trainer.callback_metrics['val_loss']:.4f}")
        # 학습 손실 출력
        if trainer.callback_metrics.get("train_loss"):
            print(f"Training Loss: {trainer.callback_metrics['train_loss']:.4f}")
        if "val_pearson" in metrics:
            print(f"Validation Pearson Correlation: {metrics['val_pearson']:.4f}")
        print("-" * 40)