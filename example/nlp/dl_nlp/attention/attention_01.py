import random
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import numpy as np

NUM_LETTERS = 8
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2

VOCAB_TOKENS = ["<pad>", "<sos>", "<eos>"] + [
    chr(ord("a") + i) for i in range(NUM_LETTERS)
]
print("\nVOCAB_TOKENS\n", VOCAB_TOKENS)

VOCAB_SIZE = len(VOCAB_TOKENS)
MIN_SEQ_LEN = 3
MAX_SEQ_LEN = 7
NUM_TRAIN_SAMPLES = 2000
NUM_VALID_SAMPLES = 200

EMBED_DIM = 32
HIDDEN_SIZE = 64
ATTN_DIM = 64
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3


class CopyDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        min_len: int,
        max_len: int,
        vocab_start: int,
        vocab_end: int,
        sos_index: int,
        eos_index: int,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.min_len = min_len
        self.max_len = max_len
        self.vocab_start = vocab_start
        self.vocab_end = vocab_end
        self.sos_index = sos_index
        self.eos_index = eos_index

        self.data = [self._make_sample() for _ in range(num_samples)]

    # 위에꺼 이용해서 아래 함수 완성
    #  return은 텐서형태의 src,target 리턴
    def _make_sample(self):
        seq_len = random.randint(self.min_len, self.max_len)
        seq = [
            random.randint(self.vocab_start, self.vocab_end - 1) for _ in range(seq_len)
        ]
        src = torch.tensor(seq, dtype=torch.long)
        tgt = torch.tensor([self.sos_index] + seq + [self.eos_index], dtype=torch.long)
        return src, tgt

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[index]


# torch.tensor랑 torch.Tensor랑 뭐가 다르지


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):

    # tag = [1,3,5,7,9,12,2]
    # trg_input = [1,3,5,7,9,12] 이건 디코더 넘기는 용
    # trg_output= [3,5,7,9,12,2] 이건 트레인 훈련 용

    src_seqs, trg_seqs = zip(*batch)

    # src_lens = [len(s) for s in src_seqs]
    # trg_lens = [len(s) for s in trg_seqs]

    # max_src_len = max(src_lens)ㅇ
    # max_trg_len = max(trg_lens)

    src_batch = pad_sequence(src_seqs, batch_first=True, padding_value=PAD_IDX)
    src_mask = (src_batch != PAD_IDX).unsqueeze(1)
    tgr_input = pad_sequence(
        [s[:-1] for s in trg_seqs], batch_first=True, padding_value=PAD_IDX
    )
    trg_output = pad_sequence(
        [s[1:] for s in trg_seqs], batch_first=True, padding_value=PAD_IDX
    )
    return src_batch, tgr_input, trg_output, src_mask


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_idx,
        )

        self.rnn = nn.GRU(
            embed_dim,
            hidden_size,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, src, src_lengths=None):
        emb = self.embedding(src)
        outputs, hidden = self.rnn(emb)
        return outputs, hidden


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size_enc, hidden_size_dec, attn_dim):
        super().__init__()
        self.W_h = nn.Linear(hidden_size_enc * 2, attn_dim, bias=False)
        self.W_s = nn.Linear(hidden_size_dec, attn_dim, bias=False)
        self.v_a = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, encoder_hidden, decoder_hidden, mask=None):
        Wh = self.W_h(encoder_hidden)
        Ws = self.W_s(decoder_hidden).unsqueeze(1)

        score = self.v_a(torch.tanh(Wh + Ws)).squeeze(-1)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.squeeze(1)
            score = score.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(score, dim=-1)

        # Context Vector: (B, 1, S) x (B, S, H*2) -> (B, 1, H*2)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_hidden)

        return context.squeeze(1), attn_weights


class DecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # RNN 입력: 임베딩 + 컨텍스트(hidden*2)
        self.rnn = nn.GRU(embed_dim + hidden_size * 2, hidden_size, batch_first=True)

        self.attention = AdditiveAttention(
            hidden_size_enc=hidden_size,  # 내부에서 *2 처리됨
            hidden_size_dec=hidden_size,
            attn_dim=ATTN_DIM,
        )
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, trg_input, encoder_outputs, encoder_mask, hidden):
        B, T_in = trg_input.size()
        emb = self.embedding(trg_input)

        outputs = []
        attn_list = []

        # Encoder Hidden (2, B, H) -> Decoder Hidden (1, B, H)로 변환
        if hidden.size(0) == 2:
            decoder_hidden = hidden[-1].unsqueeze(0)
        else:
            decoder_hidden = hidden

        curr_hidden = decoder_hidden.squeeze(0)  # (B, H)
        input_step = emb[:, 0, :]  # <sos>

        for t in range(T_in):
            context, attn_weights = self.attention(
                encoder_outputs, curr_hidden, encoder_mask
            )
            attn_list.append(attn_weights.unsqueeze(1))

            # RNN 입력 결합
            rnn_input = torch.cat(
                [input_step.unsqueeze(1), context.unsqueeze(1)], dim=-1
            )

            # RNN 실행
            out, next_hidden = self.rnn(rnn_input, curr_hidden.unsqueeze(0))

            # 결과 저장
            logits = self.fc_out(out.squeeze(1))
            outputs.append(logits.unsqueeze(1))

            # 다음 스텝 준비
            curr_hidden = next_hidden.squeeze(0)
            if t + 1 < T_in:
                input_step = emb[:, t + 1, :]

        logits_all = torch.cat(outputs, dim=1)
        attn_weights_all = torch.cat(attn_list, dim=1)

        return logits_all, attn_weights_all


class Seq2Seq(nn.Module):

    def __init__(self, encoder: Encoder, decoder: DecoderWithAttention):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg_input, src_mask):
        encoder_outputs, enc_hidden = self.encoder(src)
        logits, attn_weights = self.decoder(
            trg_input, encoder_outputs, src_mask, enc_hidden
        )

        return logits, attn_weights


def indices_to_tokens(indices: List[int]) -> List[str]:
    return [VOCAB_TOKENS[idx] for idx in indices]


def print_example(src, trg_input, trg_output, pred_indices):
    """
    src: (S,)
    trg_input: (T,)
    trg_output: (T,) # [x1, ..., xL, <eos>]
    pred_indices: (T,) # 예측 토큰 인덱스
    """
    src_tokens = indices_to_tokens(src)
    trg_tokens = indices_to_tokens(trg_output)
    pred_tokens = indices_to_tokens(pred_indices)

    print("--------------------------------------------------")
    print("\nsrc_tokens\n", src_tokens)
    print("\ntrg_tokens\n", trg_tokens)
    print("\npred_tokens\n", pred_tokens)


# 1. 훈련 데이터셋 생성(CopyDataset)
train_dataset = CopyDataset(
    num_samples=NUM_TRAIN_SAMPLES,
    min_len=MIN_SEQ_LEN,
    max_len=MAX_SEQ_LEN,
    vocab_start=3,
    vocab_end=VOCAB_SIZE,
    sos_index=SOS_IDX,
    eos_index=EOS_IDX,
)

# 2. 검증 데이터셋 생성
valid_dataset = CopyDataset(
    num_samples=NUM_VALID_SAMPLES,
    min_len=MIN_SEQ_LEN,
    max_len=MAX_SEQ_LEN,
    vocab_start=3,
    vocab_end=VOCAB_SIZE,
    sos_index=SOS_IDX,
    eos_index=EOS_IDX,
)

# 3. 훈련/검증 데이터 로더 생성
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
)

# 4. Encoder 생성
encodder = Encoder(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    hidden_size=HIDDEN_SIZE,
    pad_idx=PAD_IDX,
)

# 5. Decoder 생성
decoder = DecoderWithAttention(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    hidden_size=HIDDEN_SIZE,
    pad_idx=PAD_IDX,
)

# 6. Seq2Seq 모델 생성
model = Seq2Seq(encoder=encodder, decoder=decoder)

# 7. 손실 함수 생성
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# 8. 옵티마이저 생성
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# 9. def train_one_epoch() 생성
# - 손실을 구할때 logits(N,C), tag_output(N)을 넘김
# - 평균 손실은 토큰 단위로
# - 1 epoch 동안 (훈련(2000)/검증(200)) 같이 진행


# 9. def train_one_epoch() 생성
def train_one_epoch(epoch_idx):
    model.train()  # 모델을 학습 모드로 전환
    epoch_loss = 0

    # --- [훈련 단계] ---
    for i, batch in enumerate(train_loader):
        src, trg_input, trg_output, src_mask = batch

        optimizer.zero_grad()  # 기울기 초기화

        # 모델 순전파 (Forward)
        logits, _ = model(src, trg_input, src_mask)

        # --- [차원 맞추기] ---
        # Decoder의 반복문 구조상 logits의 길이가 trg_output보다 1 짧을 수 있습니다.
        # (마지막 <eos> 예측 전 단계까지만 돌기 때문)
        # 따라서 길이를 맞춰주기 위해 슬라이싱을 합니다.
        min_len = min(logits.size(1), trg_output.size(1))
        curr_logits = logits[:, :min_len, :]  # (B, min_len, V)
        curr_trg_output = trg_output[:, :min_len]  # (B, min_len)
        # --------------------

        # 손실 계산 (Flatten: N*T, C)
        loss = criterion(
            curr_logits.reshape(-1, VOCAB_SIZE), curr_trg_output.reshape(-1)
        )
        # 역전파 및 가중치 업데이트
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    train_avg_loss = epoch_loss / len(train_loader)

    # --- [검증 단계 (Validation)] ---
    # "훈련/검증 같이 진행" 조건에 따라 epoch마다 검증 수행
    model.eval()  # 평가 모드로 전환
    valid_loss = 0

    with torch.no_grad():  # 검증 땐 기울기 계산 안 함
        for batch in valid_loader:
            src, trg_input, trg_output, src_mask = batch

            logits, _ = model(src, trg_input, src_mask)

            # 검증 데이터도 차원 맞추기
            min_len = min(logits.size(1), trg_output.size(1))
            curr_logits = logits[:, :min_len, :]
            curr_trg_output = trg_output[:, :min_len]

            loss = criterion(
                curr_logits.reshape(-1, VOCAB_SIZE), curr_trg_output.reshape(-1)
            )
            valid_loss += loss.item()

    valid_avg_loss = valid_loss / len(valid_loader)

    # 결과 출력
    print(
        f"Epoch: {epoch_idx+1:02} | Train Loss: {train_avg_loss:.4f} | Valid Loss: {valid_avg_loss:.4f}"
    )


# --- [실제 학습 루프 실행] ---
print("Training Start...")
for epoch in range(NUM_EPOCHS):
    train_one_epoch(epoch)
