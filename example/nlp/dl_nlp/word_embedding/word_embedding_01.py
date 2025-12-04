import torch
import torch.nn as nn

vocab = {"나는": 0, "밥을": 1, "먹었다": 2, "학교에": 3, "갔다": 4}
vocab_size = len(vocab)
emb_dim = 3

embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)

print("(W 가중치 표 shape):", embed.weight)

sentance = ["나는", "밥을", "먹었다"]
idxs = [vocab[word] for word in sentance]
idx_tensor = torch.tensor(idxs)

emb_vectors = embed(idx_tensor)

print("임베딩 벡터들:\n", emb_vectors)
