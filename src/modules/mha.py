import torch
import torch.nn as nn

class MHA(nn.Module):
    def __init__(self, embedding_dim, num_heads, kv_prealloc_size=2048, device='cuda', **kwargs):
        """
        Initialize multi-head attention

        Args:
        - embedding_dim (int): the embedding dimension.
        - num_heads (int): the number of attention heads.
        - kv_prealloc_size (int): how long of a sequence is expected for the kv cache. Overestimates are preferred.
        - batch_size_alloc (int): the planned batch size.

        Returns:
        - None
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.kv_prealloc_size = kv_prealloc_size
        self.layer_idx = kwargs['layer_idx']

        self.device = device

        # intialize weight matrices
        self.qkv = nn.Linear(self.embedding_dim, 3*self.embedding_dim, dtype=torch.bfloat16)
        self.out_proj = nn.Linear(self.embedding_dim, self.embedding_dim, dtype=torch.bfloat16)

        # intialize kv cache
        self.kv_cache = torch.zeros(
            (2, self.kv_prealloc_size, self.num_heads, self.head_dim),
            device=device, dtype=torch.bfloat16
        )

        # stores how much of the kv cache has been used.
        self.seq_pos = None
    
    def forward(self, x):
        """
        Computes self-attention for the given input tensor and prepares the KV cache.

        Args:
        - x (torch.Tensor): the input tensor with shape (seq_len, embedding_dim)

        Returns:
        - values (torch.Tensor): tensor with shape (seq_len, embedding_dim) representing the self-attended values
        """

        # ----- GENERATE KV CACHE AND COMPUTE Q -----

        # prepare causal attention mask
        x = x[0] # remove batch dimension
        self.seq_pos = x.shape[0]
        causal_mask = torch.triu(torch.full((self.seq_pos, self.seq_pos), -10000.0, device=self.device), 1).unsqueeze(-1)

        # if sequence is longer than kv_cache is allocated, reallocate.
        if self.seq_pos >= self.kv_prealloc_size:
            self.kv_prealloc_size = self.seq_pos + 512
            self.kv_cache = torch.zeros(
                (2, self.kv_prealloc_size, self.num_heads, self.head_dim),
                device=self.device, dtype=torch.bfloat16
            )

        seq_qkv = self.qkv(x)
        seq_qkv = seq_qkv.reshape((-1, 3, self.num_heads, self.head_dim)).permute((1, 0, 2, 3))

        # set kv cache.
        self.kv_cache[0,:self.seq_pos] = seq_qkv[1]
        self.kv_cache[1,:self.seq_pos] = seq_qkv[2]

        # ----- DO SELF-ATTENTION -----
        seq_attention_scores = torch.einsum('bhe,she->bsh', seq_qkv[0], seq_qkv[1])
        # next we apply temperature.
        seq_attention_scores /= self.head_dim**.5

        # add causal mask.
        seq_attention_scores += causal_mask

        # print('(NHEADS, Q, KV)')
        # print('MASKED self attention:\n', seq_attention_scores.permute(2,0,1))

        # softmax
        seq_attention_scores = torch.softmax(seq_attention_scores, dim=1) # pretty sure this does the max trick internally.

        values = torch.einsum('bsh,she->bhe', seq_attention_scores, seq_qkv[2]).reshape((1, -1, self.embedding_dim))

        # ----- PROJECT CONCATENATED HEADS -----

        values = self.out_proj(values)

        return values

    def decode(self, batch, causal_mask):
        """
        This function computes query, key, value for a batch, applies attention to compute weighted values,
        and returns the combined values.

        Args:
        - batch (torch.Tensor): tensor of shape (batch_size, embedding_dim)
        - causal_mask (torch.Tensor): tensor of shape (batch_size, batch_size) used to mask the batch attention scores. (causal_mask == 0) should be lower triangular; nonzero values are of course large and negative. (-10000 recommended.)

        Returns:
        - values (torch.Tensor): tensor of shape (batch_size, embedding_dim) containing the combined values.
        """

        batch_size = batch.shape[-2]

        # ----- BEGIN GET BATCH QKV -----

        # first compute query, key, value for the batch.
        self.batch_qkv = self.qkv(batch).reshape((-1, 3, self.num_heads, self.head_dim)).permute((1, 0, 2, 3))

        # also store causal mask for later reference.
        self.last_causal_mask = causal_mask

        # ----- BEGIN CAUSAL ATTENTION SCORING -----

        # next we need to do the attention.
        attention_scores = torch.empty((batch_size, self.seq_pos+batch_size, self.num_heads), device=self.device, dtype=torch.bfloat16)
        attention_scores[:,:self.seq_pos] = torch.einsum('bhe,she->bsh', self.batch_qkv[0], self.kv_cache[0,:self.seq_pos]) # (batch, seq, nheads)
        attention_scores[:,self.seq_pos:] = torch.einsum('bhe,she->bsh', self.batch_qkv[0], self.batch_qkv[1]) # (batch, batch, nheads)

        # print('Unmasked self attention:\n', attention_scores[:,self.seq_pos:].permute(2,0,1))

        # we don't need to mask the seq attention but we do need to mask the batch attention scores.
        attention_scores[:,self.seq_pos:] += self.last_causal_mask.unsqueeze(-1)


        # ----- BEGIN SOFTMAX -----

        # next we apply temperature.
        attention_scores /= self.head_dim**.5 # this is the problem, probably

        # then softmax!
        attention_scores = torch.softmax(attention_scores, dim=1)

        # ----- BEGIN ASSEMBLING VALUES -----

        values = (
            torch.einsum('bsh,she->bhe', attention_scores[:,:self.seq_pos], self.kv_cache[1,:self.seq_pos]) +
            torch.einsum('bsh,she->bhe', attention_scores[:,self.seq_pos:], self.batch_qkv[2])
        ).reshape((1, -1, self.embedding_dim)) # reshape here is basically a concat of the heads.

        # ----- PROJECT CONCATENATED HEADS -----

        values = self.out_proj(values)

        return values

    def update_kv(self, i):
        """
        This function is called after evaluating all outputs of the batch and the best has been decided.

        Args:
        - i (int): index within the last batch which ended up being accepted.

        Returns:
        None
        """

        # figure out which elements need to be appended to the kv_cache.
        is_child = (self.last_causal_mask[i] == 0) # determine which were parents. This is lower triangular.
        # i'th row has parents of i. can only be true up to i, though.
        num_children = is_child.sum().item()
        self.kv_cache[:,self.seq_pos:self.seq_pos+num_children] = self.batch_qkv[1:,is_child]
        self.seq_pos += num_children

    def truncate_kv(self, pos):
        # no need to overwrite anything in th kv cache.
        # just mark the end of the sequence as sooner.
        self.seq_pos = pos

    # if our sequence grows beyond the original allocation, we may need to reallocate. this does that.
    # NEEDS REWRITE DUE TO NEW UPDATE_KV
    def check_realloc_kv(self):
        if self.seq_pos >= self.kv_prealloc_size:
            # new size
            new_prealloc_size = round(self.kv_prealloc_size*1.5)
            # realloc and copy
            new_kv_cache = torch.zeros((2, new_prealloc_size, self.num_heads, self.head_dim), device=self.device)
            new_kv_cache[:len(self.kv_cache)] = self.kv_cache
            # set member variables
            self.kv_cache = new_kv_cache
            self.kv_prealloc_size = new_prealloc_size

if __name__ == '__main__':
    mha = MHA(1024, 8)
    print(mha.state_dict().keys())