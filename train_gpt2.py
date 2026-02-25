from dataclasses import dataclass
from logging import config
import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attention (materializes the large (T,T) matrix  for all the queries and keys)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5)) # (B, nh, T, T)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # (B, nh, T, T)
        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of attention heads
    n_embd: int = 768 # dimensionality of the embeddings (should be divisible by n_head)

class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight # reason for doing this is that the objective of the token embedding matrix and the language modeling head matrix are the same (same semantic meaning embeddings should be close to each other in the embedding space), so it makes sense to share weights between them, and we are sharing (saving) about 30% of the parameters in the model by doing this, which is nice

        # init params
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd)
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layer norm and the classifier
        x = self.transformer.ln_f(x) # (B, T, n_embd)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        
        if targets is not None:
            # reshape so that the tokens in the vocabulary are the last dimension
            B, T, vocab_size = logits.size()
            logits = logits.view(B*T, vocab_size)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # negative log likelihood loss 
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from HuggingFace."""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}, "model_type must be one of 'gpt2', 'gpt2-medium', 'gpt2-large', or 'gpt2-xl'"
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        print(f"Loading weights from pretrained gpt: {model_type}")
        
        # n_layer, n_head, n_embd are determined by the model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768), # 124M parameters
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024), # 355M parameters
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280), # 774M parameters
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600), # 1558M parameters
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer since it's not a learned parameter and will be re-created by the CausalSelfAttention module
        
        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these since they're not learned parameters and will be re-created by the CausalSelfAttention module
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # ignore these since they're not learned parameters and will be re-created by the CausalSelfAttention module
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear layer, so we need to strip off the "Conv1D" part of the name when loading the weights
        # this means we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        
        return model

# ----------------------------------------------------------------

import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
    
        # at init load tokens from the disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Loaded {len(self.tokens)} tokens from the dataset")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")
        
        # state
        self.current_position = 0
        
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B*T + 1] # (B * T)
        x = buf[:-1].view(B, T) # (B, T)
        y = buf[1:].view(B, T) # (B, T)
        # advance the current position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + B*T + 1 >= len(self.tokens):
            self.current_position = 0
        return x, y

# ----------------------------------------------------------------

# attempt to auto-detect the device
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "cpu"
else:
    device = "cpu"
print(f"Using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available(): 
    torch.cuda.manual_seed(1337)

# Fix for MPS: ensure float32 precision (MPS has issues with mixed precision)
if device == "mps":
    torch.set_default_dtype(torch.float32)

# get a data batch
train_loader = DataLoaderLite(B=4, T=1024)

torch.set_float32_matmul_precision('high') # for faster training on float32 (this is a no-op on other devices, but on mps/amp it can give a significant speedup, and on newer NVIDIA GPUs with TF32 support it can also give a speedup with minimal loss of precision) (upto 8x speedup on MPS which is basically free (no loss of precision))


num_return_sequences = 5
max_length = 30

model = GPT(config=GPTConfig())
model.to(device)
model = torch.compile(model) # best way and should be used as default always for kernel fusion (minimize round trips between GPU and HBM (GPU memory)) so it makes training significantly faster, especially for smaller models where the forward pass is not very expensive, but we have to do it many times, so the overhead of the Python interpreter becomes significant, and torch.compile helps to eliminate that overhead by fusing the kernels together and minimizing the number of round trips between the GPU and the CPU (where the Python interpreter is running), which can give a significant speedup (up to 10x or more) for training small to medium sized models, and even for larger models it can still give a speedup, but it's not as dramatic since the forward pass is more expensive and dominates the runtime anyway, but it's still recommended to use torch.compile for all training runs since it's basically free and can only help with performance)
# optimize!
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    # with torch.autocast(device_type=device, dtype=torch.bfloat16): # Use autocast for bfloat16 precision (newer and bettter, as in fp16 we need to use scalers etc that increase complexity)
    #     logits, loss = model(x, y)
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step {i}: loss {loss.item()}")
# print(logits.shape)
# print(logits.dtype)
# at random initialization, we want each token probability to be roughly uniform, so the loss should be close to -log(1/vocab_size)
# print(loss)

import sys; sys.exit(0)

model.eval()
model.to(device)

# prefix tokens
import tiktoken
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to('cuda')

# generate! right now x is (B, T) where B = 5, T = 8
# set the seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits for the next token
    logits = model(x) # (B, T, vocab_size)
    # take the logits at the last position
    logits = logits[:, -1, :] # (B, vocab_size)
    # apply softmax to get probabilities
    probs = F.softmax(logits, dim=-1) # (B, vocab_size)
    # do top-k sampling of 50 (huggingface pipeline default)
    # topk_probs here becomes (5, 50), topk_indices is (5, 50)
    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
    # select a token from the top-k probabilities
    ix = torch.multinomial(topk_probs, num_samples=1) # (B, 1)
    # gather the corresponding indices
    xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
    # append to the sequence
    x = torch.cat((x, xcol), dim=1) # (B, T+1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)