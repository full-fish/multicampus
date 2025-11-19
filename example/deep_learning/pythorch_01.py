import torch

# print("PyTorch version:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())
# print("Device:", "cuda" if torch.cuda.is_available() else "cpu")

# # x = torch.tensor([[1, 2, 3]])
# # print("----")
# # print(x)
# # print(x.shape)
# # print(x.dtype)
# # print(x.dim())
# # print("----")

# # x = torch.arange(0, 10)
# # print(x.shape)
# # print("zero:\n", torch.zeros((2, 3)))
# # print("ones:\n", torch.ones((3, 3)))
# # print("rand:\n", torch.rand((2, 2, 2)))
# from torch.utils.data import TensorDataset, DataLoader

# X = torch.randn(100, 1)  # 100개 샘플
# y = 3 * X + 1 + 0.1 * torch.randn_like(X)
# dataset = TensorDataset(X, y)
# loader = DataLoader(dataset, batch_size=16, shuffle=True)
# for xb, yb in loader:
#     print("batch X:", xb.shape, "batch y:", yb.shape)
#     break

from torch.utils.data import TensorDataset, DataLoader

X = torch.randn(100, 1)  # 100개 샘플
y = 3 * X + 1 + 0.1 * torch.randn_like(X)
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)
for xb, yb in loader:
    print("batch X:", xb.shape, "batch y:", yb.shape)
    break


from torch.utils.data import Dataset


class CSVDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.X = torch.tensor(df[["feature1", "feature2"]].values, dtype=torch.float32)
        self.y = torch.tensor(df["target"].values, dtype=torch.float32)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]


dataset = CSVDataset("data.csv")
loader = DataLoader(dataset, batch_size=32, shuffle=True)
