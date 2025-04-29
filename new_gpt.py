# new_gpt.py

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


@dataclass
class ModelConfig:
    batch_size: int = 1654
    block_size: int = 32
    n_embd: int = 32
    n_head: int = 2
    n_layer: int = 2
    dropout: float = 0.0
    learning_rate: float = 1e-3
    max_iters: int = 5000
    eval_interval: int = 100
    eval_iters: int = 50
    device: torch.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    file_name: str = "input_childSpeech_trainingSet.txt"


class BaseModule(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class Head(BaseModule):
    def __init__(self, config: ModelConfig, head_size: int):
        super().__init__(config)
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(BaseModule):
    def __init__(self, config: ModelConfig, num_heads: int, head_size: int):
        super().__init__(config)
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(BaseModule):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(BaseModule):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        head_size = config.n_embd // config.n_head
        self.sa = MultiHeadAttention(config, config.n_head, head_size)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(BaseModule):
    def __init__(self, config: ModelConfig, vocab_size: int):
        super().__init__(config)
        self.vocab_size = vocab_size

        self.token_embedding_table = nn.Embedding(vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size)

        self.apply(self._init_weights)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.config.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# Helper functions for data processing
def load_data(config: ModelConfig):
    with open(config.file_name, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data, encode, decode, vocab_size


def get_batch(config: ModelConfig, split: str, train_data: torch.Tensor, val_data: torch.Tensor):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i + config.block_size] for i in ix])
    y = torch.stack([data[i + 1:i + config.block_size + 1] for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y


@torch.no_grad()
def estimate_loss(model: GPTLanguageModel, config: ModelConfig, train_data: torch.Tensor, val_data: torch.Tensor):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(config, split, train_data, val_data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train_model(config: ModelConfig):
    torch.manual_seed(1337)

    # Initialize tracking lists
    train_losses = []
    val_losses = []

    # Load and process data
    train_data, val_data, encode, decode, vocab_size = load_data(config)

    # Initialize model
    model = GPTLanguageModel(config, vocab_size).to(config.device)
    print(f'{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters')

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Training loop
    for iter in range(config.max_iters):
        if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            losses = estimate_loss(model, config, train_data, val_data)
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch(config, 'train', train_data, val_data)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return model, encode, decode, {
        'train_losses': train_losses,
        'val_losses': val_losses
    }


# Usage example
if __name__ == "__main__":
    config = ModelConfig()

    model, encode, decode = train_model(config)

    # Generate text
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))