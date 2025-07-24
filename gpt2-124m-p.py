from transformers import GPT2LMHeadModel
from torch.nn import functional as F
from dataclasses import dataclass
import torch.nn as nn
import torch

# Device selection
device_type = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ------------------------------------------

class CasualSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Key, Query, Value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Output Projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # Regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Not really a 'bias', more of a mask, but following the OpenAI/HF naming
        self.register_budder("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size() # Batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate Query, Key, Values for all heads in batch and move head forward to be the batch dim
        # nh = number of heads. hs = head size. C = nh * hs
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Attention (Materializes the large (T, T) matrix for all the queries and keys)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v # (B, nh, T, T) x  (B, nh, T, hs) ->  (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # Re-assemble all head outputs side be side
        # Output Projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024 # Max Sequence Length
    vocab_size: int = 50257 # GPT 2 Tokenizer has length of 50,257. 256 raw byte tokens + 50,000 merges + 1 special token
    n_layer: int = 12 # Number of layers
    n_head: int = 12  # Number of heads
    n_embd: int = 768 # Embedding Dimension


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    @classmethod
    def from_pretrained(cls, model_type):
        # Loads pretrained GPT-2 model weights from huggingface
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        print("Loading weights from pretrained GPT: %s" % model_type)

        # n_layer, n_head, and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768), # 124M Params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), # 350M Params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280), # 774M Params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600), # 1558M Params
        }[model_type]

        config_args['vocab_size'] = 50257 # Always 50,257 for gpt model checkpoints
        config_args['block_size'] = 1024 # Always 1,024 for gpt model checkpoints

        # Create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] #Disgard this mask / buff, not a param

        # Init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Copt whule ensuring that all parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # Ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # Ignore these, just a buffer
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # Basically the OpenAI checkpoints use a "Conv1D" module, byt we inly want to use a vanilla linear. This means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Special treatment for the Conv1D weights we need to transpose
                assert sd[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # Start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.names_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Create optim groups. Any parameters tha are 2D will be weight decayed, otherwise no
        # Ex. all weighgt tensors in matmuls + embeddings decay, all biases and layernorms don't
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"

        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

