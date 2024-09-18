from src.tokenizing.tokenizing import tokenizing
import pandas as pd
def preprocessing(data, model_name):
    # 안쓰는 컬럼을 삭제합니다.
    data = data.drop(columns=['id'])
    # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
    try:
        targets = data['label'].values.tolist()
    except:
        targets = []

    # 텍스트 데이터를 전처리합니다.
    inputs = tokenizing(data, model_name)

    return inputs, targets

def augmentation(data):
    augmented_data = data.copy()
    non_zero_labels = augmented_data[augmented_data['label'] != 0]
    non_zero_labels[['sentence_1', 'sentence_2']] = non_zero_labels[['sentence_2', 'sentence_1']]
    augmented_data = pd.concat([augmented_data, non_zero_labels], ignore_index=True)
    return augmented_data