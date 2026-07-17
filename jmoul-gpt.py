"""
A minimal, single-file GPT (decoder-only transformer) for character-level
language modeling.
 
----------------------------------------------------------------------------
Setup
----------------------------------------------------------------------------
    pip install torch
        For an RTX 5090 (Blackwell / sm_120) you need a recent PyTorch built
        against CUDA 12.8. Grab the exact command from https://pytorch.org
        (it looks like:  pip install torch --index-url https://download.pytorch.org/whl/cu128 ).
        An older torch will fail at runtime with "no kernel image available".
 
Data (tiny Shakespeare, ~1 MB) -- download next to this file:
    curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
 
Run:
    python jmoul-gpt.py
"""



from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

#Config data class
@dataclass
class GPTConfig:
    block_size: int = 256 # Context length, length of single block that runs thru transformer during training
    vocab_size: int = None # Fill in after
    n_layer: int = 6 #How many transformer blocks total
    n_head: int = 6 # How many attention heads
    n_embd: int = 384 #Number of dimensions per token vector (must be divisible by n_head)
    dropout: int = 0.2 # dropout prob for regularization
    
    
    
class CausalSelfAttention(nn.Module):
    #"Self"   -> Q, K, V are all computed from the same input x.
    # "Causal" -> a token may only attend to itself and earlier tokens.
    
    #In this implementation we will calculate Q, K, V and multi head split
    # ourselves just for visibility into how this is setup.
    # Offload the actual calculations to pytorch
    
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 # Needs to be 0 to do MHA
        
        self.n_head = config.n_head # Number of attention heads
        self.n_embd = config.n_embd # Number of dimensions in token vector
        self.head_dim = config.n_embd // config.n_head # Size of each heads slice of token vector
        
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.dropout_p = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        
        
    def forward(self, x):
        B, T, C = x.shape
        
        # B = batch size, num of sequences of tokens in a batch
        # T = Number of tokens per sequence, CONTEXT WINDOW!!
        # C = # of dimensions per token vector
        
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        #SPLITTING THE Q, K, V VECTORS IN HEAD FOR MHA
        # vvv AI Generated vvv
        # --- split into heads -------------------------------------------------
        # Two moves turn each token's 384-vector into 6 independent heads:
        #   1) .view     regroups the C = 384 channels as (n_head, head_dim) = (6, 64).
        #                No data is moved -- 384 is just reinterpreted as 6 * 64.
        #   2) .transpose(1, 2) swaps the token axis and the head axis, so each
        #                head becomes its own (T, head_dim) attention problem and
        #                (B, n_head) ride along as leading "batch" dimensions.
        #
        #   start             (B, T, C)                = (64, 256, 384)
        #   after .view       (B, T, n_head, head_dim) = (64, 256,  6,  64)
        #   after .transpose  (B, n_head, T, head_dim) = (64,  6,  256, 64)
        #
        # Now the innermost length-64 row = one token's q/k/v in one head, and a
        # (T, head_dim) = (256, 64) slice = that head's Q/K/V matrix from the paper.
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1,2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1,2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1,2)
        
        # Actually run attention, calculating scaled dot product.
        
        # Actual attention part, dot product time.
        # Dot product = calculating how "aligned" two vectors are.
        # AKA, measures directional alignment scaled by magnitude
        # Score gets driven up by pointing in the SAME direction, and by HOW MUCH they are pointing in same dir
        
        # Negative values = activly misaligned
        # Positive values = More aligned, higer = more aligned
        
        # q and k are used first to compute scores then scaled and softmax to get WEIGHTS
        # then matmul with V is applied to each. V is the "ACTUAL CONTENT" being mixed
        # Q and K only decide how much of V to take
        # weights (result of Q and K) @ V is weighted sum. Scaling each V vector by weight and adding up
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        
        # --- recombine heads --------------------------------------------------
        # Undo the split: (B, nh, T, head_dim) -> (B, T, nh, head_dim) -> (B, T, C).
        # .contiguous() is needed because transpose only changes the view, and
        # .view() requires the underlying memory to be laid out contiguously.
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection - Allows all the heads to mix the context/features they learn
        out = self.resid_dropout(self.proj(out))
        
        # Return the tensor x
        return out
    
    
class MLP(nn.Module):
    """Position-wise feed-forward network. Runs on each token independently."""
    #https://arxiv.org/pdf/2012.14913
    # MLP is where a lot of facts get stored
    def __init__(self, config):
        super().__init__()
        # Standard transformer recipe: expand to 4x width, apply a nonlinearity,
        # then project back down. This is where most of the model's parameters live.
        
        # Lift 384-dim token vector to 1536-dim space. Allows for more "info" to get stored in
        # within the learned weights at very high dimensionality
        self.fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        
        # We multiply by 4 just as a naturally found good tradeoff for compute vs performance
        # Nothing inherantly dervied that 4x, it's just found to be a good value
        
        #Project back down to 384-dim vector.
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd) 
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc(x)
        x = F.gelu(x)        # smooth nonlinearity; GPT-2 uses GELU rather than ReLU
        x = self.proj(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):
    """One transformer block: attention sub-layer + MLP sub-layer, each residual."""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Run normalized x through attention block
        # We add x instead of replacing because we want the result of attn to be a MODIFICATION
        # of x. We want "here's the changes we need to make to x", NOT
        # "here's the new x"

        # if we did just x = self.attn, we could be deleting useful information that had
        # previously been learned and stored in x
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    
    
class GPT(nn.Module):
    """The full model: embeddings -> stack of blocks -> projection to vocab logits."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embedding, turning token IDs into the vectors (384 in our case)
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Position embedding, here we are using LEARNED position embedding, diff that
        # sinusodial used in og attention is all you need. no diff in accuracy basically
        
        #The position gets ADDED to the token vector. The network LEARNS to use
        # the position information added to the vector during training.
        #this is a bit confusing, I will need to watch videos on this and read
        # more on it. but seems easier than using the sincos positing.
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
        # Setup transformer blocks. Weights are stored in the blocks!
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        
        # Final layer norm, same deal as in attention blocks.
        # subtract mean of all values, divide by the std deviation, so mean is 0 and variance is 1
        # We also apply two LEARNABLE params to PER dimension
        # y (gain) and shift (b) with each being length 384 so the layer
        # can re-stretch and re-shift each dimension that is useful
        # FORUMULA: output = y(gain) * normalized_vector * b(shift)
        # WATCH VIDEO ON THIS TO LEARN MORE (GET BETTER VISUALIZATION)
        self.ln_f = nn.LayerNorm(config.n_embd)
        # Mapping each positions vector to a score for every token in the vocab...
        # here is where we actually predict the next token :O
        
        # lm_head = LANGUAGE MODEL HEAD, output layer that produces the predictions
        # Goes Linear(384, 65) (65 = one per vocab character)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Apply is basically going thru each "module" (like MLP, attn block, layer norm)
        # and initilizing the weights for each module.
        self.apply(self._init_weights)
    
    # module = the current module that is being recusivly traversed to initilize weights
    # also setting std dev to 0.02 as that is what was done in gpt2 paper.
    def _init_weights(self, module):
        if (isinstance(module, nn.Linear)):
            # module.weight returns the weight matrix
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                # In pytorch, a training _ modifies the module IN PLACE meaning
                # it will operate directly on the object in memory instead of returning
                # a new tensor or something. this will not return anything useful.
                # I.e, if we ran nn.init.zeros(1000), it would return a tensor of 0s
                nn.init.zeros_(module.bias)
            # Seperate if block here because we do NOT want to apply bias to embedding 
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
    def forward(self, idx, targets=None):
        """
        idx -> (B, T integers), tensor, will get used to create x. 64 sequences * 256 tokens
        B -> Batch size = 64, how many independent sequences we process at once (purely training thruput dims)
        T -> Time + sequence length, how many tokens in each sequence. Equal to block size which is context length
        token_emd -> (B, T, 384), same B and T but we added 384 content/feature vectors (n_embd)
        pos -> (T,) integers [0, 255] (256) on T axis, list of positions of tokens in block
        pos_emd -> (T, 384) location/position vectors, T positions but now each is a 384-dim vector
        token_emd + pos_emd -> (B, T, 384). pos_emd has no  B axis, pytorch adds same 256 position vectors to each of the 64 sequences
        x = dropout(...) -> (B, T, 384), this is the finished tensor to go thru the blocks
        """
        
        # In python, (T,) the comma indicates a 1D shape 
        
        # B = vocab, sequences of a batch
        # T = 256, block size, arb value of how many tokens we look at once
        B, T = idx.shape
        device = idx.device
        
        # (B, T, n_embd)
        token_emd = self.token_embedding(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        pos_emd = self.position_embedding(pos)
        x = self.dropout(token_emd + pos_emd)
        
        # For future add th e RoPE absolute positioning here
        
        # Run blocks
        # This is the ACTUAL thinking/learning part, running x thru the blocks over and over
        # x is constantly reassigned to new value, but we do add to x in the block forward pass
        # so essentially it's x + x after each run, not just block run then set x to new block output
        for block in self.blocks:
            x = block(x)
        # Final layernorm. Ensuring the logits are reasonable when output
        x = self.ln_f(x)
        
        # Projection to the vocab logits, raw scores for each possible token output
        # So the result here is a tensor (B (batch size), T(time/seq_len), 65(vocab_size))
        logits = self.lm_head(x)
        
        # Loss starts off as None and loss is ONLY COMPUTED when targets is set
        # This is because during training we are calculating loss, however, during inference
        # we do not calculate the loss
        loss = None
        if targets is not None:

            """
            In this function, we are basically passing in the models prediction for what comes after index i,
            then targets is giving us the actual value of what comes after index i
            Cross entropy is returning a number corresponding to HOW WRONG the model prediction was. use to update
            weights. 
            Here's what the cross_entropy function is doing:

            Step 1: Run softmax function on the raw logits. This will convert the logits from just raw numerical scores
            into a probability distribution, all positive values adding up to 1. Higher logit score = higher probability

            Step 2: Get the negative log likelyhood. In this step the cross_entropy function gets the actual score
            the model assigned to the correct token (targets[i]) and takes the negative log of it to get the 
            loss score. loss_for_position_i = -log( p[ correct_char ] )

            """
            loss = F.cross_entropy(
                # (B*T, vocab_size), 64*256 = 16384 predictions
                # Each "prediction" is a row of 65 numbers (logits) corresponding to next token (based on pos)
                logits.view(-1, logits.size(-1)),  
                # (16384,) tuple, .view(-1) is flattening the target tensor into 1d
                targets.view(-1),
            )

        return logits, loss
    
    # Do not track calculations done since this function is purely forward
    # pass and we are just using it for generating next token, not going to use
    # anything for backprop. 
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # idx = Tensor (B, T) the starting context, sample one new token at a time and append
        # to this.
        # Since we are generating a single sequence, idx = (1, T), and the goal is to calculate (1,T+1)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            # Self is model instance here since we are still in GPT class
            # loss is no op
            logits, _ = self(idx_cond) # (B, T, vocab_size) returned from forward()
            # logits starting out is (B, T, 65) which is the prediction for next token at every position
            # However we only care about the next token position for the next token to ADD
            # : Keep all B sequences, -1 take the last time-step, : keep all 65 logits
            logits = logits[:, -1, :] / temperature #keeping only the last step (B, vocab)
            # Since we have (B, vocab), that is bascially like (1, 65) with 65 corresponding to logits for next token

            # Top K filtering is we 0 out all tokens except the k top scores
            # so we dont sample some random bullshit (if we had like a high temp, meaning every value gets more likely)
            # Optional top-k filtering: zero out everything except the k most likely
            # tokens so we never sample something wildly improbable.
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # v[-1] broadcasts the tensor so it compares lowest (last pos) logit score
                # to each score in the og logits tensor. Set to 0 if lower
                logits[logits < v[:, [-1]]] = float("-inf")

            # Running softmax on the 65 logits that correspond to each of the chars in vocab
            # at this point, we already broke down the tensor to the last time-slice's 65 logits with logits[:, -1, :],
            # meaning this is predicting the next token in the series (as calculated on line 279)

            # -1 = vocab dimension, dim param tells us what dimension to run the calculations on.
            # We always want to run the softmax on the VOCAB dimension to get the prob over next token.
            # The reason we use dim=-1 is because that just corresponds to last axis in tensor, and
            # vocab will always be the final axis/dimension since it's the feature dimension. 
            # Amount of dimensions in tensor can change so using a static value like 2 
            # could result in selecting the wrong axis
            probs = F.softmax(logits, dim=-1) # (B, vocab), B = 1

            # Random sampling from the probability distribution above
            # Each index's chance of being picked is equal to the probability assigned above by softmax
            # This is finally where we pick the next character that the model predicts.
            # Shape of (1, 1)
            next_id = torch.multinomial(probs, num_samples=1) # sample: (B, 1)

            # torch.cat joins tensors along the given dimension.
            # [idx, next_id] is the list of tensors to join along dimension 1 which in our
            # case is the time dimension -> appending the token to time dimension is
            # essentially steping forward one turn T+1
            idx = torch.cat([idx, next_id], dim=1)
        return idx
    
# ============================================================================
# DATA/TOKENIZATION (Character Level Tokenization)
# ============================================================================
# Character Level - Tokens are just individual characters from input txt
# Vocab is the set of unique chars within the file/input data
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Chars is a deduped list of every character in input sorted alpha-numerically
chars = sorted(list(set(text)))
vocab_size = len(chars)

print(f"Vocab Size: {vocab_size}")

# String to index, dictionary mapping char to index int
stoi = {}
# Index to string, mapping index int to corresponding char
itos = {}
for i, ch in enumerate(chars):
    stoi[ch] = i
    itos[i] = ch

def encode(input_string):
    """
    Convert (or "encode") input string into token IDs
    ex, "Hello" -> [int, int, int, int, int] 
    """
    return [stoi[c] for c in input_string]

def decode(token_ids):
    """
    Decode a list of token_ids into a readable String
    ex, [5, 2, 3, 3, 11] -> "Hello"
    """
    return "".join(itos[id] for id in token_ids)

# Input data is getting preped as a very long 1d tensor of token IDs
data = torch.tensor(encode(text), dtype=torch.long)

#Retain the last 10% of input data to do eval on hahaha
n_resv_tokens = int(0.9 * len(data))

# 90% of data for training
train_data = data[:n_resv_tokens]

# 10% of data for eval
eval_data = data[n_resv_tokens:]

# ============================================================================
# 4. Training setup
# ============================================================================
# Hyperparams
batch_size = 64 # How many sequences of tokens we train on at once (B, T, n_embd)
max_iters = 50000 # Total number of opitimization steps
eval_interval = 500 # eval and print every this amount of steps
eval_iters = 200 # Average the loss over this many batches per eval
learning_rate = 3e-4 # AdamW step size

device = "cuda" if torch.cuda.is_available else "cpu"

torch.manual_seed(69420)

# Build the model then send to GPU
# Set the vocab size here cuz we only know it after we import data
config = GPTConfig(vocab_size=vocab_size)
model = GPT(config).to(device)
print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

# Get one batch of data to eval/train on. Dictated by batch_size
def get_batch(split):
    source = train_data if split == "train" else eval_data
    
    # Subtract block_size to ensure that anywhere we start we do not overrun the selection
    # Gives us a 1D tensor of 
    ix = torch.randint(len(source) - config.block_size, (batch_size,))
    
    # i is a random start position to begin the sequence of tokens.
    # We start at i, then iterate forward to build the sequence of block_size tokens
    # we do this ix times (which is equal to batch_size) so we have 
    # an input tensor of (batch_size, block_size)
    x = torch.stack([source[i : i + config.block_size] for i in ix])
    # Get the same window except add 1 to it so we have the target
    # This is equal to the previous stack except shifted by one.
    # We do not need the first token since every token is just predicting
    # it's next token.
    y = torch.stack([source[i + 1: i + 1 + config.block_size] for i in ix])
    
    return x.to(device), y.to(device)

    """
    In estimate_loss() we run on both training data and eval data to do a compairson
    between the model's calculated loss for eval and train data which are kept seperate.
    We never pass in the eval data to the model during training. We would expect to see
    that the loss goes down for both - if only the training data loss goes down but the
    eval data loss does not, then we read that as the model is just "remembering" the
    training data and not actually learning. This compairson is to detect overfitting
    in the model.
    """
@torch.no_grad()
def estimate_loss():
    # Estimates the loss over several batches
    
    out = {}
    model.eval() # Put model in eval mode so there's no dropouts
    
    for split in ["train", "eval"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            # x = Input tokens, tensor (batch_size, block_size)
            # y = x shifted by 1, true next token (used to calculate loss in model)
            # Again, one "batch" is a tensor of (64, 256) which in english means
            # 64 sequences of 256 tokens. Randomly sampled using randint in get_batch
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item() # Get the actual float from the tensor (loss is current a 0d tensor)
        out[split] = losses.mean() # Average loss over the 200 samples
    model.train() # Back to training mode so dropouts happen
    return out

# ============================================================================
# 5. Training loop
# ============================================================================

import os
from datetime import datetime

# Create a new uniquely-named file for THIS training run (once, before the loop)
os.makedirs("samples", exist_ok=True)  # put them in a samples/ folder; drop this line to use cwd
run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
sample_file = f"samples/training_samples_{run_timestamp}.txt"
print(f"Writing generation samples to: {sample_file}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # Perodically print the loss for both train and eval so we can monitor progress
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"On step: {iter} -- Train loss: {losses['train']} || Eval Loss: {losses['eval']}")
    
    # Write to a file every 500 steps
    if iter % 500 == 0:
        # Generate from the model at its current state
        context = torch.zeros((1, 1), dtype=torch.long, device=device)  # start token (id 0)
        generated_ids = model.generate(context, max_new_tokens=500)[0].tolist()
        sample_text = decode(generated_ids)

        # Append to the run's file (mode "a" = append, so each checkpoint adds on)
        with open(sample_file, "a", encoding="utf-8") as f:
            f.write(f"Text gen step {iter}:\n{sample_text}\n\n")
            f.write(f"Losses {iter}:\n{losses}\n\n")
            f.write("=" * 60 + "\n\n")  # separator between checkpoints        
    
    # Start optimization step
    xb, yb = get_batch("train") # Get sample batch of TRAIN data
    logits, loss = model(xb, yb) # Run the batch through the model to get LOSS and LOGITS
    # In pytorch, gradients pile on top of eachother if you do not explicitly reset them
    # In "normal training" we always want to reset them in between runs
    optimizer.zero_grad(set_to_none=True)
    
    loss.backward() # Run backprop algorithm to calculate new gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    print(f"Step: {iter} complete")
    
# ============================================================================
# 6. Generate some text from the trained model
# ============================================================================
model.eval() # Again, disable dropout
context = torch.zeros((1, 1), dtype=torch.long, device=device)

generated = model.generate(context, max_new_tokens=500)[0].tolist()
print(decode(generated))