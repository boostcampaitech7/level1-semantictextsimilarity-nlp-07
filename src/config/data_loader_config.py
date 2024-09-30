
DATA_LOADER_CONFIG = { 
  "batch_size" : 16,
  "shuffle" : True,
  "train_path" : './data/train_after_hanspell (1).csv',
  "dev_path" : './data/dev_after_hanspell.csv',
  "test_path" : './data/dev_after_hanspell.csv',
  "predict_path" : './data/test_after_hanspell.csv',
  "output_path" : './data/sample_submission.csv'  
}

OPTIMIZER_CONFIG = {
  "learning_rate": 1e-5,
  "max_epoch": 50
}

TRAIN_INPUT_FEATURES = ['input_ids', 'attention_mask', 'label']

TEST_INPUT_FEATURES = ['input_ids', 'attention_mask']