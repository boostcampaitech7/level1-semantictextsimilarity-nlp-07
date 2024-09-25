from pytorch_lightning.callbacks import EarlyStopping

class EarlyStopping(EarlyStopping):
    def __init__(
        self,
        monitor = 'val_pearson',
        patience = 7,
        verbose = True,
    ):
        super().__init__(
            monitor=monitor,
            patience= patience,
            verbose=verbose,
        )