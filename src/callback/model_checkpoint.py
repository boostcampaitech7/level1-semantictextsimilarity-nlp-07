from pytorch_lightning.callbacks import ModelCheckpoint
class ModelCheckpoint(ModelCheckpoint):
    def __init__(
        self,
        monitor='val_pearson',
        mode='max',
        save_top_k=1,
        save_last=False,
        model_name = None,
        filename='best-{model_name}-{epoch:02d}',
        verbose=True,
        dirpath='./checkpoints'
    ):
        if model_name:
            dirpath = f'./{model_name}/checkpoints'
            
        super().__init__(
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            save_last=save_last,
            filename=filename,
            verbose=verbose,
            dirpath=dirpath
        )
                     