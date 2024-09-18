from tqdm.auto import tqdm
from transformers import AutoTokenizer

def tokenizing(dataframe, model_name):
    data = []
    tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=130)
    for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
        # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
        text = '[SEP]'.join([item[text_column] for text_column in ['sentence_1', 'sentence_2']])
        outputs = tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
        data.append(outputs['input_ids'])
    return data