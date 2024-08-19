# %% [markdown]
# # Importing Dataset

# %%
# import os

# %%
# def extract():
#     text = ''
#     path = '/kaggle/input/marvel-cinematic-universe-dialogue-dataset'
#     for file in os.listdir(path):
#         path_file = os.path.join(path + '/', file)
#         with open(path_file, 'r', errors= 'ignore') as f:
#             text += f.read()
#     return text

# # %%
# text = extract()

# %%
# text[:100]

# %% [markdown]
# # Text Processing

# %%
class Preprocess():
    def __init__(self, text):
        super().__init__()
        self.text = text
        
    def create_vocab(self):
        vocab = sorted(list(set(self.text)))
        self.stoi = {s: i for i, s in enumerate(vocab)}
        self.itos = {i: s for s, i in self.stoi.items()}
        return vocab, len(vocab), self.stoi, self.itos
    
    def encode(self, string):
        return [self.stoi[char] for char in string]
    
    def decode(self, array):
        return ''.join(self.itos[idx] for idx in array)

# %%
# text_processor = Preprocess(text)
# vocab, vocab_size, stoi, itos = text_processor.create_vocab()

# # %%
# print(text_processor.encode('hello'))
# text_processor.decode(text_processor.encode('hello'))

# %% [markdown]
# # Set Device

# %%
import torch

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

# %% [markdown]
# # Split Dataset

# %%
import torch.nn as nn

# %%
# data = torch.tensor(text_processor.encode(text), dtype = torch.long)
# data[:50], len(data)

# # %%
# n = int(0.9 * len(data))
# train = data[:n]
# val = data[n:]
# len(train), len(val)

# %%
# def split(type):
#     data = train if type == 'train' else val
#     idx = torch.randint(len(data) - block_size, (batch_size, ))
#     X = torch.stack([data[i: i + block_size] for i in idx])
#     y = torch.stack([data[i + 1: i + block_size + 1] for i in idx])
#     X, y = X.to(device), y.to(device)
#     return X, y

# %%
batch_size = 64 
block_size = 256 
max_iters = 3000
eval_interval = 500
learning_rate = 1e-3
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
vocab_size = 84

stoi = {'\n': 0,
 ' ': 1,
 '!': 2,
 '"': 3,
 '$': 4,
 '%': 5,
 '&': 6,
 "'": 7,
 '(': 8,
 ')': 9,
 '*': 10,
 ',': 11,
 '-': 12,
 '.': 13,
 '/': 14,
 '0': 15,
 '1': 16,
 '2': 17,
 '3': 18,
 '4': 19,
 '5': 20,
 '6': 21,
 '7': 22,
 '8': 23,
 '9': 24,
 ':': 25,
 ';': 26,
 '?': 27,
 'A': 28,
 'B': 29,
 'C': 30,
 'D': 31,
 'E': 32,
 'F': 33,
 'G': 34,
 'H': 35,
 'I': 36,
 'J': 37,
 'K': 38,
 'L': 39,
 'M': 40,
 'N': 41,
 'O': 42,
 'P': 43,
 'Q': 44,
 'R': 45,
 'S': 46,
 'T': 47,
 'U': 48,
 'V': 49,
 'W': 50,
 'X': 51,
 'Y': 52,
 'Z': 53,
 'a': 54,
 'b': 55,
 'c': 56,
 'd': 57,
 'e': 58,
 'f': 59,
 'g': 60,
 'h': 61,
 'i': 62,
 'j': 63,
 'k': 64,
 'l': 65,
 'm': 66,
 'n': 67,
 'o': 68,
 'p': 69,
 'q': 70,
 'r': 71,
 's': 72,
 't': 73,
 'u': 74,
 'v': 75,
 'w': 76,
 'x': 77,
 'y': 78,
 'z': 79,
 '¡': 80,
 'à': 81,
 'é': 82,
 'ñ': 83}

itos = {0: '\n',
 1: ' ',
 2: '!',
 3: '"',
 4: '$',
 5: '%',
 6: '&',
 7: "'",
 8: '(',
 9: ')',
 10: '*',
 11: ',',
 12: '-',
 13: '.',
 14: '/',
 15: '0',
 16: '1',
 17: '2',
 18: '3',
 19: '4',
 20: '5',
 21: '6',
 22: '7',
 23: '8',
 24: '9',
 25: ':',
 26: ';',
 27: '?',
 28: 'A',
 29: 'B',
 30: 'C',
 31: 'D',
 32: 'E',
 33: 'F',
 34: 'G',
 35: 'H',
 36: 'I',
 37: 'J',
 38: 'K',
 39: 'L',
 40: 'M',
 41: 'N',
 42: 'O',
 43: 'P',
 44: 'Q',
 45: 'R',
 46: 'S',
 47: 'T',
 48: 'U',
 49: 'V',
 50: 'W',
 51: 'X',
 52: 'Y',
 53: 'Z',
 54: 'a',
 55: 'b',
 56: 'c',
 57: 'd',
 58: 'e',
 59: 'f',
 60: 'g',
 61: 'h',
 62: 'i',
 63: 'j',
 64: 'k',
 65: 'l',
 66: 'm',
 67: 'n',
 68: 'o',
 69: 'p',
 70: 'q',
 71: 'r',
 72: 's',
 73: 't',
 74: 'u',
 75: 'v',
 76: 'w',
 77: 'x',
 78: 'y',
 79: 'z',
 80: '¡',
 81: 'à',
 82: 'é',
 83: 'ñ'}

# %%
# Xtr, ytr = split('train')
# Xtr.shape, ytr.shape

# %% [markdown]
# # Define Error List

# %%
import torch.nn.functional as F

# %%
# @torch.no_grad()
# def estimate_loss():
#     out = {}
#     model.eval()
#     for splits in ['train', 'val']:
#         losses = torch.zeros(eval_iters)
#         for k in range(eval_iters):
#             X, Y = split(splits)
#             logits, loss = model(X, Y)
#             losses[k] = loss.item()
#         out[splits] = losses.mean()
#     model.train()
#     return out

# %% [markdown]
# # Create Model

# %%
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

# %%
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# %%
class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# %%
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# %%
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
#             print(self(idx_cond))
#             print("idx_cond", idx_cond)
            logits, loss = self(idx_cond)
            
            logits = logits[:, -1, :] # becomes (B, C)
            
            probs = F.softmax(logits, dim=-1) # (B, C)
#             print("probs", probs)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# %%
# model = BigramLanguageModel()
# m = model.to(device)
# print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# %% [markdown]
# # Train the Model

# %%
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# for iter in range(max_iters):
#     if iter % eval_interval == 0 or iter == max_iters - 1:
#         losses = estimate_loss()
#         print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
#     xb, yb = split('train')
#     logits, loss = model(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()

# %% [markdown]
# # Generate Text

# %%
# def generate(num_words):
#     context = torch.zeros((1, 1), dtype=torch.long, device=device)
#     print(text_processor.decode(m.generate(context, max_new_tokens = num_words)[0].tolist()))

# # %%
# generate(1000)

# # %%
# model_save_path = 'bigram_language_model.pth'
# torch.save(model.state_dict(), model_save_path)

# %%



