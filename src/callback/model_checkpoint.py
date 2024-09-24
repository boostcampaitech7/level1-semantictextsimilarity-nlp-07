from pytorch_lightning.callbacks import ModelCheckpoint
class ModelCheckpoint(ModelCheckpoint):
    def __init__(
        self,
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=False,
        model_name=None,
        verbose=True,
        dirpath='./checkpoints'
    ):
        filename = 'best-{epoch:02d}-{val_loss:.2f}'
        
        if model_name:
            dirpath = f'./{model_name}/checkpoints'
            filename = f'best-{model_name}-{{epoch:02d}}-{{val_loss:.2f}}'
            
        super().__init__(
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            save_last=save_last,
            filename=filename,
            verbose=verbose,
            dirpath=dirpath
        )
                    