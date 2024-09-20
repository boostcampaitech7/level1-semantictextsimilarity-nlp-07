import pandas as pd
from transformers import AutoTokenizer


def preprocessing(data: pd.DataFrame, model_name: str) -> pd.DataFrame:
    # 안쓰는 컬럼을 삭제합니다.
    data = data.drop(columns=['id'])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    tokenized_output = data.apply(
        lambda row: tokenizer(row['sentence_1'], row['sentence_2'],
                              add_special_tokens=True, padding='max_length', 
                              truncation=True, max_length=128), axis=1
    )
    
    data['input_ids'] = tokenized_output.apply(lambda x: x['input_ids'])
    data['attention_mask'] = tokenized_output.apply(lambda x: x['attention_mask'])
        
    return data

def augmentation(data: pd.DataFrame):
    augmented_data = data.copy()
    non_zero_labels = augmented_data[augmented_data['label'] != 0]
    non_zero_labels[['sentence_1', 'sentence_2']] = non_zero_labels[['sentence_2', 'sentence_1']]
    augmented_data = pd.concat([augmented_data, non_zero_labels], ignore_index=True)
    return augmented_data