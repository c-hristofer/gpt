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
batch_size      = 4 # How many independent sequences will be processed in paralel
block_size      = 8 # THe maximum context length for predictions
max_iters       = 100000
eval_interval  = 300
learning_rate   = 1e-2
device          = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters      = 200
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
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):

        # idx and targets are both(B,T) tensor of integers
        logits = self.token_embedding_table(idx)                # (B,T,C) This is a Tensor with Batch, Time, Channel 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)                            # Turn it into 2D Array
            targets = targets.view(B*T)                             # Turn it into 1D Array
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            
            # Get the predictions
            logits, loss = self(idx)
            # Focus only on the last time step
            logits = logits[:, -1, :]                           # Becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)                    # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)             # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# Create a PyTorch optimizer using AdamW optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # Every once in a while, evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    # Sample a batch of data
    xb, yb = get_batch('train')
    # Evaluate loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))