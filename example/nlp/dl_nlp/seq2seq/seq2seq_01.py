import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

chars = list("abcdefghijklmnopqrstuvwxyz")

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

itos = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN] + chars
print("\nitos\n", itos)

stoi = {ch: i for i, ch in enumerate(itos)}
print("\nstoi\n", stoi)

PAD_IDX = stoi[PAD_TOKEN]
SOS_IDX = stoi[SOS_TOKEN]
EOS_IDX = stoi[EOS_TOKEN]

vocab_size = len(itos)


# random.randit(), random.choice(), 단어 하나 리턴
def random_string(min_len=3, max_len=7):
    return "".join(random.choices(chars, k=random.randint(min_len, max_len)))


# 문자열(단어)를 인덱스로 바꾸고 앞뒤로 <sos>, <eos>를 붙여서 리턴
def encode_sequence(text: str):
    return torch.tensor(
        [SOS_IDX] + [stoi.get(ch, stoi[PAD_TOKEN]) for ch in text] + [EOS_IDX],
        dtype=torch.long,
    )


# 인덱스 벡터가 입력되면 <sos>, <eos>를 지우고 인덱스에 해당하는 문자를 붙여서 문자열(단어)를 리턴
def decode_sequence(indices):
    return "".join(
        [itos[idx] for idx in indices if idx not in (SOS_IDX, EOS_IDX, PAD_IDX)]
    )


text = random_string()
print("\ntext\n", text)

encoded_text = encode_sequence(text)
print("\nencoded_text\n", encoded_text)

decoded_text = decode_sequence(encoded_text)
print("\ndecoded_text\n", decoded_text)


class ReverseDataset(Dataset):
    def __init__(self, num_samples=2000, min_len=3, max_len=7):
        super().__init__
        self.data = []
        for _ in range(num_samples):
            s = random_string(min_len, max_len)
            input = encode_sequence(s)
            target = encode_sequence(s[::-1])  # 역순
            self.data.append((input, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 패딩 처리 하는 함수
def collate_fn(batch):

    inputs, targets = zip(*batch)

    max_input_len = max([len(seq) for seq in inputs])
    max_target_len = max([len(seq) for seq in targets])

    padded_inputs = [seq + [PAD_IDX] * (max_input_len - len(seq)) for seq in inputs]
    padded_targets = [seq + [PAD_IDX] * (max_target_len - len(seq)) for seq in targets]

    return (
        torch.tensor(padded_inputs, dtype=torch.long),
        torch.tensor(padded_targets, dtype=torch.long),
    )


# # train_loader: PyTorch DataLoader로, 학습용 배치(예: (src_batch, tgt_batch))를 순회해서 제공합니다.
# # 보통은 DataLoader(..., batch_size=..., shuffle=True, collate_fn=...) 같이 생성하고
# # 학습 루프에서 `for src, tgt in train_loader:` 형태로 사용합니다.
# # 주의: 시퀀스 길이가 가변이면 collate_fn으로 패딩 처리 필요, GPU 전달은 .to(device)로 수행.
train_dataset = ReverseDataset(num_samples=2000)
train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
)


class Encoder(nn.Module):
    # 모델 초기화 함수: 어휘 크기, 임베딩 차원, 은닉 상태 차원을 인수로 받습니다.
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()

        # 1. 임베딩(Embedding) 계층: 입력 토큰 ID를 밀집된 벡터로 변환합니다.
        #    vocab_size: 어휘의 총 개수
        #    embed_dim: 임베딩 벡터의 차원
        #    padding_idx=PAD_IDX: 패딩 토큰(PAD_IDX)은 0 벡터로 임베딩되어 학습에 기여하지 않게 합니다.
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim=embed_dim, padding_idx=PAD_IDX
        )

        # 2. GRU(Gated Recurrent Unit) 계층: 시퀀스 데이터를 처리하는 순환 신경망입니다.
        #    embed_dim: 입력 특성 차원 (임베딩 차원)
        #    hidden_size=hidden_dim: 은닉 상태 벡터의 차원
        #    batch_first=True: 입력 텐서의 형태가 [배치 크기, 시퀀스 길이, 특성 차원]임을 의미합니다.
        self.gru = nn.GRU(
            embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

    # 순전파(Forward Pass) 함수: 인코딩을 수행합니다.
    def forward(self, src):
        # src: 입력 텐서. [batch_size, src_len] (예: [32, 10])

        # 1. 임베딩 계층 통과
        embedded = self.embedding(
            src
        )  # [batch_size, src_len, embed_dim] (예: [32, 10, 256])

        # 2. GRU 계층 통과
        # outputs: 시퀀스의 각 타임 스텝의 출력. (사용되지 않을 수도 있지만, 어텐션에 필요)
        # hidden: 최종 은닉 상태. 이 상태가 디코더의 초기 은닉 상태로 사용됩니다.
        outputs, hidden = self.gru(embedded)

        # outputs: [batch_size, src_len, hidden_dim]
        # hidden: [num_layers * num_directions, batch_size, hidden_dim] (단방향, 단일 계층이므로 [1, batch_size, hidden_dim])
        return outputs, hidden


class Decoder(nn.Module):
    # 초기화 함수: 인코더와 유사하며, 최종적으로 토큰을 예측하기 위한 선형 계층이 추가됩니다.
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()  # super().init() 대신 super().__init__()을 사용하는 것이 일반적입니다.

        # 1. 임베딩(Embedding) 계층: 출력 토큰 임베딩을 담당합니다.
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim=embed_dim, padding_idx=PAD_IDX
        )

        # 2. GRU(Gated Recurrent Unit) 계층
        self.gru = nn.GRU(
            embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # 3. 출력 선형 계층 (Fully Connected Layer):
        #    GRU의 은닉 상태를 어휘 크기로 투영하여 다음 토큰에 대한 로짓(logits, 확률 분포 이전의 값)을 계산합니다.
        #    hidden_dim: GRU 출력의 차원
        #    vocab_size: 예측해야 할 어휘의 총 개수
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    # 순전파 함수: 한 타임 스텝의 디코딩을 수행합니다.
    # input_step: 현재 타임 스텝의 입력 토큰 [batch_size] (예: [32])
    # hidden: 이전 타임 스텝의 은닉 상태 [1, batch_size, hidden_dim] (예: [1, 32, 512])
    def forward(self, input_step, hidden):

        # 1. 텐서 형태 변환: 시퀀스 길이를 나타내는 차원(seq_len=1)을 추가합니다.
        input_step = input_step.unsqueeze(
            1
        )  # [batch_size] => [batch_size, 1] (예: [32, 1])

        # 2. 임베딩 계층 통과
        embedded = self.embedding(
            input_step
        )  # [batch_size, 1, embed_dim] (예: [32, 1, 256])

        # 3. GRU 계층 통과: 현재 임베딩과 이전 은닉 상태를 입력합니다.
        output, hidden = self.gru(embedded, hidden)

        # output: GRU의 현재 타임 스텝 출력. [batch_size, 1, hidden_dim] (예: [32, 1, 512])
        # hidden: 업데이트된 은닉 상태. [1, batch_size, hidden_dim] (예: [1, 32, 512])

        # 4. 출력 텐서 형태 변환: seq_len=1 차원을 제거합니다.
        output = output.squeeze(
            1
        )  # [batch_size, 1, hidden_dim] => [batch_size, hidden_dim] (예: [32, 512])

        # 5. 선형 계층 통과: 로짓 계산
        logits = self.fc_out(
            output
        )  # [batch_size, hidden_dim] => [batch_size, vocab_size] (예: [32, 6000])

        # logits: 다음 토큰에 대한 예측 확률 분포의 로그 형태 (로짓)
        # hidden: 다음 타임 스텝으로 전달될 업데이트된 은닉 상태
        return logits, hidden


class Seq2Seq(nn.Module):
    # 모델 초기화 함수: 인코더와 디코더 인스턴스를 인수로 받습니다.
    def __init__(self, encoder, decoder):
        super().__init__()

        # 인코더와 디코더 모듈을 클래스 변수로 저장합니다.
        self.encoder = encoder
        self.decoder = decoder

    # 순전파(Forward Pass) 함수
    # src: 소스(입력) 시퀀스. [batch_size, src_len]
    # tgt: 타겟(정답 출력) 시퀀스. [batch_size, tgt_len]
    # teacher_forcing_rate: 티처 포싱 비율. 0.7이면 70% 확률로 정답 토큰을 사용합니다.
    def forward(self, src, tgt, teacher_forcing_rate=0.7):
        # 1. 배치 크기와 타겟 시퀀스 길이 추출
        batch_size = src.size(0)  # 배치 크기 (예: 32)
        tgt_len = tgt.size(1)  # 타겟 시퀀스 길이 (예: 15)

        # 2. 최종 출력을 저장할 텐서 초기화
        # outputs: [batch_size, tgt_len, vocab_size]. 디코더의 모든 타임 스텝 예측 로짓을 저장합니다.
        # 주의: 'vocab_size'는 모델 클래스 외부에서 정의되거나 인수로 전달되어야 합니다.
        outputs = torch.zeros(
            batch_size, tgt_len, vocab_size
        )  # vocab_size는 실제 환경에 맞게 조정 필요

        # 3. 인코더 실행 및 초기 은닉 상태 획득
        # _: 인코더의 전체 출력 (outputs). 여기서는 사용하지 않습니다.
        # hidden: 인코더의 최종 은닉 상태. 디코더의 초기 은닉 상태로 사용됩니다.
        _, hidden = self.encoder(src)

        # 4. 디코더의 첫 번째 입력 토큰 설정
        # 타겟 시퀀스의 첫 번째 토큰인 <sos> (Start Of Sequence) 토큰으로 디코딩을 시작합니다.
        input_step = tgt[
            :, 0
        ]  # [batch_size]. 모든 배치에 대해 <sos> 토큰을 가져옵니다.

        # 5. 디코더 루프 실행 (t=1 부터 시작)
        # t=1은 타겟 시퀀스의 첫 번째 실제 단어 위치(두 번째 토큰)에 해당합니다.
        for t in range(1, tgt_len):
            # 5-1. 디코더 실행
            # logits: 다음 토큰에 대한 예측 로짓. [batch_size, vocab_size]
            # hidden: 업데이트된 은닉 상태. 다음 스텝으로 전달됩니다.
            logits, hidden = self.decoder(input_step, hidden)

            # 5-2. 현재 타임 스텝의 예측 로짓 저장
            outputs[:, t, :] = (
                logits  # [batch_size, vocab_size] 로짓을 outputs의 t번째 슬롯에 저장
            )

            # 5-3. 티처 포싱 여부 결정
            # teacher_forcing_rate 확률로 True가 됩니다.
            teacher_force = random.random() < teacher_forcing_rate

            # 5-4. 모델의 예측 토큰 (top1) 획득
            # 로짓에서 가장 높은 값을 가진 단어의 인덱스를 찾습니다. (argmax)
            top1 = logits.argmax(dim=1)

            # 5-5. 다음 입력 토큰 설정 (티처 포싱 로직)
            if teacher_force:
                # 티처 포싱이 적용될 경우: 다음 타임 스텝의 '정답' 토큰을 입력으로 사용합니다. (학습에 도움)
                input_step = tgt[:, t]
            else:
                # 티처 포싱이 적용되지 않을 경우: 모델이 '직전에 예측한' 토큰을 다음 입력으로 사용합니다. (추론과 유사)
                input_step = top1

        # 최종 출력: [batch_size, tgt_len, vocab_size]. 디코더가 예측한 각 타임 스텝의 로짓 모음입니다.
        return outputs


embedded_dim = 64
hidden_dim = 256
num_epochs = 20

encoder = Encoder(vocab_size=vocab_size, embed_dim=embedded_dim, hidden_dim=hidden_dim)
decoder = Decoder(vocab_size=vocab_size, embed_dim=embedded_dim, hidden_dim=hidden_dim)
model = Seq2Seq(encoder, decoder)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for src_batch, tgt_batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        optimizer.zero_grad()

        outputs = model(src_batch, tgt_batch, teacher_forcing_rate=0.7)

        outputs_reshape = outputs[:, 1:, :].reshape(-1, vocab_size)
        tgt_reshape = tgt_batch[:, 1:].reshape(-1)

        loss = criterion(outputs_reshape, tgt_reshape)  # (N, C) , (N,)

        loss.backward()
        optimizer.step()

        valid_tokens = (tgt_reshape != PAD_IDX).sum().item()
        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens

    avg_loss = total_loss / total_tokens
    print(f"[Epoch {epoch:02d}] Train Loss: {avg_loss:.4f}")


def predict(model, s, max_len=20):
    model.eval()
    with torch.no_grad():
        src = encode_sequence(s).unsqueeze(0)  # [1, src_len]

        _, hidden = model.encoder(src)

        input_step = torch.tensor([SOS_IDX])  # [1]
        outputs = []

        for _ in range(max_len):
            logits, hidden = model.decoder(input_step, hidden)
            top1 = logits.argmax(dim=1)
            if top1.item() == EOS_IDX:
                break
            outputs.append(top1.item())
            input_step = top1

        pred_str = decode_sequence(outputs)
        return pred_str
