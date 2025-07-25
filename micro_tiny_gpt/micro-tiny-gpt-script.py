#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.nn import functional as F

#-------------------------------------------------------------------------------
# Tiny GPT Implemenation
#-------------------------------------------------------------------------------

# Hyperparameters
batch_size      = 64 # How many independent sequences will be processed in paralel
block_size      = 256 # THe maximum context length for predictions
max_iters       = 3000
eval_interval   = 500
learning_rate   = 3e-4
device          = 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters      = 200
n_embd          = 384
n_head          = 6
n_layer         = 6
dropout         = 0.1
warmup_steps    = 1000            # LR warm‑up duration (updates)
#--------------------

torch.manual_seed(1337)

# Read in tiny shakespeare
with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# All the unique characters that occur in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# Create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s:         [stoi[c] for c in s]  # Encoder: Takes a string and outputs a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # Decoder: Takes a lsit of integers and outputs a string

# Encoding the entire text dataset and storing it into a torch.tensor
data = torch.tensor(encode(text), dtype=torch.long)
# Split into train, test, validation sets
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# Data Loading
def get_batch(split):
    # Generates a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[  i  :  i+block_size  ] for i in ix])
    y = torch.stack([data[ i+1 : i+block_size+1 ] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            if device == 'mps':
                with torch.autocast('mps', dtype=torch.float16):
                    logits, loss = model(X, Y)
            else:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# One head of self attention
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C= x.shape
        k = self.key(x)     # (B, T, C)
        q = self.query(x)   # (B, T, C)

        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # Perform the weighted aggregation of the values
        v = self.value(x)   # (B, T, C)
        out = wei @ v       # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


# Multiple heads of self-attention in parallel
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


# A simple Linear layer followed by non-linearity
class FeedForward(nn.Module):
    
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


# Transformer Block: Communication followed by computation
class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        # n_embd: Embedding dimension, n_head: the number of heads I would like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # Final Layer Norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            
            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]

            # Get the predictions
            logits, loss = self(idx_cond)

            # Focus only on the last time step
            logits = logits[:, -1, :]  # Becomes (B, C)

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)  # (B, C)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both(B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C) This is a Tensor with Batch, Time, Channel
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb   # (B, T, C)
        x = self.blocks(x)      # (B, T, C)
        x = self.ln_f(x)        # (B, T, C)
        logits = self.lm_head(x)# (B, T, vocab_size)


        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)                            # Turn it into 2D Array
            targets = targets.view(B * T)                             # Turn it into 1D Array
            loss = F.cross_entropy(logits, targets)

        return logits, loss


model = GPTLanguageModel()
m = model.to(device)

# Create a PyTorch optimizer using AdamW optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Cosine decay schedule that starts *after* the warm‑up
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max_iters - warmup_steps,
    eta_min=1e-5,
)

for iter in range(max_iters):

    # Every once in a while, evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    # Sample a batch of data
    xb, yb = get_batch('train')
    # Evaluate loss
    if device == 'mps':
        with torch.autocast('mps', dtype=torch.float16):
            logits, loss = m(xb, yb)
    else:
        logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)  # grad clip
    optimizer.step()

    # ----- learning‑rate schedule -----
    if iter < warmup_steps:
        lr_scale = (iter + 1) / warmup_steps
        for g in optimizer.param_groups:
            g["lr"] = learning_rate * lr_scale
    else:
        scheduler.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
# Generate a large chunk of text and trim to 1,000 lines
generated_ids = m.generate(context, max_new_tokens=40000)[0].tolist()
generated_text = decode(generated_ids)
lines = generated_text.splitlines()
# If we have fewer than 1,000 lines, pad with empty lines
if len(lines) < 1000:
    lines += [""] * (1000 - len(lines))
final_text = "\n".join(lines[:1000])
print(final_text)
with open("generated-shakespeare2.txt", "w") as f:
    f.write(final_text)