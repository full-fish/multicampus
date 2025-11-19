from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import torch.nn as nn

x = torch.linspace(0, 1, steps=100).unsqueeze(1)
y = 3 * x + 1 + 0.1 * torch.randn_like(x)

x_train, x_test = x[:80], x[80:]
y_train, y_test = y[:80], y[80:]

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)


class MLPRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)


model = MLPRegressor()

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 200

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()  # ① 이전 단계의 gradient 초기화
        pred = model(xb)  # ② 입력 xb를 통해 모델의 예측값 계산
        loss = criterion(pred, yb)  # ③ 예측값(pred)과 실제값(yb) 간 손실 계산
        loss.backward()  # ④ 손실을 기준으로 각 파라미터의 gradient 계산 (역전파)
        optimizer.step()  # ⑤ 계산된 gradient를 이용해 파라미터 업데이트
        running_loss += loss.item() * xb.size(0)

    epoch_loss = running_loss / len(train_dataset)
    if (epoch + 1) % 20 == 0:
        print(f"[Epoch {epoch+1}/{num_epochs}] Train MSE: {epoch_loss:.4f}")

    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        for xb, yb in test_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            total_loss += loss.item() * xb.size(0)

        test_loss = total_loss / len(test_dataset)
        print(f"\n[Test MSE] : {test_loss:.4f}")
