import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[], sentence1=None, sentence2=None):
        self.inputs = inputs
        self.targets = targets
        self.sentence1 = sentence1  # 문장 1
        self.sentence2 = sentence2  # 문장 2

    def __getitem__(self, idx):
        input_tensor = torch.tensor(self.inputs[idx])

        if self.sentence1 is not None and self.sentence2 is not None:
            s1 = self.sentence1[idx]
            s2 = self.sentence2[idx]

            # 타겟 데이터가 있으면 타겟도 반환
            if len(self.targets) > 0:
                target_tensor = torch.tensor(self.targets[idx])
                return input_tensor, target_tensor, s1, s2
            # 타겟 데이터가 없으면 문장 쌍만 반환
            else:
                return input_tensor, s1, s2
        else:
            if len(self.targets) > 0:
                return input_tensor, torch.tensor(self.targets[idx])
            else:
                return input_tensor

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.inputs)
