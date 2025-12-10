import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

chars = list("abcdefghijklmnopqrstuvwxyz")
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

itos = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN] + chars
stoi = {ch: i for i, ch in enumerate(itos)}

PAD_IDX = stoi[PAD_TOKEN]
SOS_IDX = stoi[SOS_TOKEN]
EOS_IDX = stoi[EOS_TOKEN]

vocab_size = len(stoi)


def random_string(
    min_len=3, max_len=7
):  # random.randint(), random.choice(), 단어하나 리턴
    length = random.randint(min_len, max_len)
    s = "".join(random.choice(chars) for _ in range(length))
    return s


def encode_sequence(text: str):  # "aaeifd"
    # 문자열(단어) 인덱스로 바꾸고 앞뒤로 <sos>, <eos>를 붙여서 리스트 리턴 [1, 4,7,11, 2]
    seq = [SOS_IDX] + [stoi[s] for s in text] + [EOS_IDX]
    return torch.tensor(seq, dtype=torch.long)


def decode_sequence(indices):
    # 인덱스 벡터가 입력되면 <sos>, <eos>를 지우고 인덱스에 해당하는 문자를 붙여서 문자열(단어)를 리턴
    result = []
    for idx in indices:
        ch = itos[idx]
        if ch in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]:
            continue
        result.append(ch)

    return "".join(result)


class ReverseDataset(Dataset):
    def __init__(self, num_samples=2000, min_len=3, max_len=5):
        super().__init__()
        self.data = []
        for _ in range(num_samples):
            s = random_string(min_len=min_len, max_len=max_len)
            input = encode_sequence(s)
            target = encode_sequence(s[::-1])
            self.data.append((input, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def collate_fn(batch):  #
    inp_seqs, tgt_seqs = zip(*batch)

    inp_lens = [len(s) for s in inp_seqs]
    tgt_lens = [len(s) for s in tgt_seqs]

    max_inp = max(inp_lens)
    max_tgt = max(tgt_lens)

    padded_inp = []
    padded_tgt = []

    for inp, tgt in zip(inp_seqs, tgt_seqs):

        pad_len_inp = max_inp - len(inp)
        padded_inp.append(
            torch.cat([inp, torch.full((pad_len_inp,), PAD_IDX, dtype=torch.long)])
        )

        pad_len_tgt = max_tgt - len(tgt)
        padded_tgt.append(
            torch.cat([tgt, torch.full((pad_len_tgt,), PAD_IDX, dtype=torch.long)])
        )

    batch_inp = torch.stack(padded_inp, dim=0)
    batch_tgt = torch.stack(padded_tgt, dim=0)

    return batch_inp, batch_tgt


train_dataset = ReverseDataset(num_samples=2000)
train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
)


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim=embed_dim, padding_idx=PAD_IDX
        )
        self.gru = nn.GRU(embed_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim=embed_dim, padding_idx=PAD_IDX
        )
        self.gru = nn.GRU(
            input_size=embed_dim, hidden_size=hidden_dim, batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_step, hidden):
        input_step = input_step.unsqueeze(1)  # (32,) =>(32,1)
        embedded = self.embedding(
            input_step
        )  # (batch, seq_len) => (batch, seq_len, hidden_size)
        output, hidden = self.gru(embedded, hidden)
        # output : (batch, seq_len, hidden) : (32, 1, hidden_size)
        # hidden : (1, batch, hidden) : (1, 32, hidden_size)

        output = output.squeeze(1)  # (batch, hidden_size)
        logits = self.fc_out(output)  # (batch, vocab_size)
        return logits, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, teacher_forcing_rate=0.7):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)

        outputs = torch.zeros(batch_size, tgt_len, vocab_size)

        _, hidden = self.encoder(src)
        input_step = tgt[:, 0]

        for t in range(1, tgt_len):
            logits, hidden = self.decoder(input_step, hidden)
            outputs[:, t, :] = logits  # (batch, vocab_size)
            teacher_force = random.random() < teacher_forcing_rate

            top1 = logits.argmax(dim=1)  # (batch, 1)

            if teacher_force:
                input_step = tgt[:, t]
            else:
                input_step = top1

        return outputs


embedded_dim = 64
hidden_dim = 256
num_epochs = 30

encoder = Encoder(vocab_size=vocab_size, embed_dim=embedded_dim, hidden_dim=hidden_dim)
decoder = Decoder(vocab_size=vocab_size, embed_dim=embedded_dim, hidden_dim=hidden_dim)
model = Seq2Seq(encoder, decoder)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for src_batch, tgt_batch in train_loader:
        optimizer.zero_grad()

        outputs = model(
            src_batch, tgt_batch, teacher_forcing_rate=0.8
        )  # (batch_size, tgt_len, vocabsize)
        # (N, C) , (N,)
        outputs_reshape = outputs[:, 1:, :].reshape(-1, vocab_size)
        tgt_reshape = tgt_batch[:, 1:].reshape(-1)

        loss = criterion(outputs_reshape, tgt_reshape)  # (N, C) , (N,)

        loss.backward()
        optimizer.step()

        valid_tokens = (tgt_reshape != PAD_IDX).sum().item()
        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens

    avg_loss = total_loss / total_tokens
    print(f"Epoch : {epoch} - loss {avg_loss:.4f}")


def predict(model, s, max_len=20):
    model.eval()
    with torch.no_grad():
        src = encode_sequence(s).unsqueeze(0)  # (1, src_len)

        _, hidden = model.encoder(src)

        input_step = torch.tensor([SOS_IDX])
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


test_sample = ["abede", "xyz", "hello", "korea"]

for s in test_sample:
    pred = predict(model, s)
    print(f"input : ", s)
    print("target : ", {s[::-1]})
    print("pred : ", pred)
