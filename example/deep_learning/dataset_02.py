"""
SV 파일을 읽어 feature와 target을 PyTorch 텐서로 변환하고,
Dataset과 DataLoader를 이용해 학습용 미니배치 단위로 데이터를 공급하는 구조를 구현"""

from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd


class CSVDataset(Dataset):
    def __init__(self, csv_file):
        super().__init__()
        df = pd.read_csv(csv_file)
        self.x = torch.tensor(df[["close", "volume"]].values, dtype=torch.int64)
        self.y = torch.tensor(df["close"].values, dtype=torch.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


dataset = CSVDataset(
    "/Users/choimanseon/Documents/multicampus/ML/example/deep_learning/stock_daily.csv"
)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for xb, yb in loader:
    print("batch X: ", xb.shape, "batch y: ", yb.shape)
    print("xb:\n", xb)
    print("yb:\n", yb)
