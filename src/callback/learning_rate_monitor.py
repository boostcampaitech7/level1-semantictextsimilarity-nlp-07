from pytorch_lightning.callbacks import LearningRateMonitor
class LearningRateMonitor(LearningRateMonitor):
    def __init__(
        self,
        logging_interval = 'step'
    ):
        super().__init__(logging_interval=logging_interval)
