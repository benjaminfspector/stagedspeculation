import logging
logger = logging.getLogger(__name__)

import math
import re
from functools import partial

from collections import namedtuple, OrderedDict
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from pprint import pprint

import sys
sys.path.append('..')
from modules import Block, GPT2Embeddings

from transformers import GPT2Config
from transformers.utils import WEIGHTS_NAME, WEIGHTS_INDEX_NAME
from transformers.utils import is_remote_url
from transformers.modeling_utils import load_state_dict
from transformers.utils.hub import cached_file, get_checkpoint_shard_files

def state_dict_from_pretrained(model_name, device=None, dtype=None):
    # If not fp32, then we don't want to load directly to the GPU
    mapped_device = 'cpu' if dtype not in [torch.float32, None] else device
    is_sharded = False
    resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                        _raise_exceptions_for_missing_entries=False)
    if resolved_archive_file is None:
        resolved_archive_file = cached_file(model_name, WEIGHTS_INDEX_NAME,
                                            _raise_exceptions_for_missing_entries=False)
        if resolved_archive_file is not None:
            is_sharded = True
    if resolved_archive_file is None:
        raise EnvironmentError(f"Model name {model_name} was not found.")
    if is_sharded:
        # resolved_archive_file becomes a list of files that point to the different
        # checkpoint shards in this case.
        resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
            model_name, resolved_archive_file
        )
        state_dict = {}
        for sharded_file in resolved_archive_file:
            state_dict.update(torch.load(sharded_file, map_location=mapped_device))
    else:
        state_dict = torch.load(cached_file(model_name, WEIGHTS_NAME), map_location=device)
    # Convert dtype before moving to GPU to save memory
    if dtype is not None:
        state_dict = {k: v.to(dtype=dtype) for k, v in state_dict.items()}
    state_dict = {k: v.to(device=device) for k, v in state_dict.items()}
    return state_dict

class GPTPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, GPT2Config):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `GPT2Config`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    @classmethod
    def from_pretrained(cls, model_name, config, *args, strict=True, device=None, dtype=None,
                        world_size=1, rank=0, **kwargs):
        """
        Instantiate a GPTPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.
        """
        # Instantiate model.
        model = cls(config, *args, device=device, dtype=dtype, **kwargs)
        # Load state_dict in cpu because we already initialized the model in GPU, and we don't
        # want extra stuff taking up more GPU memory
        state_dict = state_dict_from_pretrained(
            model_name, device='cpu', dtype=dtype
        )
        for k in list(state_dict.keys()):
            if k.endswith('attn.masked_bias'):
                del state_dict[k]
            elif k.startswith('transformer.'):
                state_dict[k[len('transformer.'):]] = state_dict[k]
                del state_dict[k]
        if model_name.rfind('/') != -1:
            model_name = model_name[model_name.rfind('/')+1:]
        # pprint(list(state_dict.keys()))
        if model_name.startswith('gpt'):
            state_dict = remap_state_dict_gpt2(state_dict, config)
        else:
            raise NotImplementedError(f'Model {model_name} not supported')
        # if world_size > 1:
        #     state_dict = shard_state_dict_tp(state_dict, config, world_size, rank)
        # pprint(list(state_dict.keys()))
        load_return = model.load_state_dict(state_dict, strict=strict)
        logger.info(load_return)
        return model

class GPTModel(GPTPreTrainedModel):

    def __init__(self, config: GPT2Config, device=None, dtype=None):
        super().__init__(config)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.sequence_parallel = getattr(config, 'sequence_parallel', True)
        assert config.activation_function in ['gelu', 'gelu_new', 'gelu_fast', 'gelu_approx',
                                              'relu', 'sqrelu']
        pad_vocab_size_multiple = getattr(config, 'pad_vocab_size_multiple', 1)
        vocab_size = (math.ceil(config.vocab_size / pad_vocab_size_multiple)
                      * pad_vocab_size_multiple)
        # TD [2022-07-30]: Force residual in fp32, seems to make fp16 training more stable
        self.residual_in_fp32 = getattr(config, 'residual_in_fp32', False)
        # These 2 options are for OPT-350m
        self.prenorm = getattr(config, 'prenorm', True)
        word_embed_proj_dim = getattr(config, 'word_embed_proj_dim', None)

        self.embeddings = GPT2Embeddings(
            config.hidden_size, vocab_size, config.max_position_embeddings,
            word_embed_proj_dim=word_embed_proj_dim, **factory_kwargs
        )

        # We change the order of dropout, residual and layer norm:
        # Instead of LN -> Attn / MLP -> Dropout -> Add, we do:
        # Dropout -> Add -> LN -> Attn / MLP, returning both the residual branch (output of Add) and
        # the main branch (output of MLP). The model definition is unchanged, but the mapping of the
        # nn.Dropout probabilities are changed.
        # This is for performance reason: we can fuse dropout + add + layer_norm.
        mha_kwargs = {}
        self.layers = nn.ModuleList([
            Block(config.n_embd, config.n_head, layer_idx=i, mha_kwargs=mha_kwargs)
            for i in range(config.num_hidden_layers)
        ])

        # Final layernorm
        self.ln_f = nn.LayerNorm(config.n_embd, dtype=torch.bfloat16)

    def forward(self, input_ids):
        hidden_states = self.embeddings(input_ids, position_ids=None)
        hidden_states = hidden_states.type(torch.bfloat16)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.ln_f(hidden_states)

    def build_decoder_batch(self, batch_tree):
        """
        Example batch tree:
        {
            2495: {8773: {43870: {}}},
            18222: {}
        }
        """
        seq_pos = self.layers[0].mha.seq_pos
        if seq_pos is None:
            raise ValueError('Seq pos is undefined -- call forward with a prompt first!')
        def get_ids_with_parents(d, idx=0, parentlist=[]):
            k_idx = {}
            for k in d.keys(): # be breadth first to ensure attention is lower triangular.
                k_idx[k] = idx
                yield (idx, k, parentlist)
                idx += 1
            for k, v in d.items():
                for ans in get_ids_with_parents(v, idx=idx, parentlist=parentlist+[k_idx[k]]):
                    yield ans
                    idx = ans[0]+1
        def nested_length(d):
            return len(d) + sum((nested_length(x) for x in d.values()))
        batch_size = nested_length(batch_tree)
        # do everything on CPU to start, which I expect to be faster.
        input_ids = np.zeros((1,batch_size), dtype=np.int32)
        causal_mask = np.full((batch_size, batch_size), -10000)
        position_ids = np.zeros((1,batch_size), dtype=np.int32)
        for i, iid, parents in get_ids_with_parents(batch_tree):
            input_ids[0,i] = iid # set the actual input id.
            position_ids[0,i] = seq_pos+len(parents) # position in sequence
            for p in parents:
                causal_mask[i,p] = 0 # allow attention to parent.
            causal_mask[i,i] = 0 # allow attention to self.
        # now we can move these tensors to the GPU.
        input_ids = torch.tensor(input_ids, device=0, dtype=torch.long)
        causal_mask = torch.tensor(causal_mask, device=0, dtype=torch.bfloat16)
        position_ids = torch.tensor(position_ids, device=0, dtype=torch.long)
        return input_ids, position_ids, causal_mask
    
    def build_decoder_listbatch(self, listbatch):
        """
        Example batch list:
        [
            (2495,)
            (2495, 8773),
            (2495, 8773, 43870),
            (18222,)
            (18222, 22)
        ]
        """
        seq_pos = self.layers[0].mha.seq_pos
        if seq_pos is None:
            raise ValueError('Seq pos is undefined -- call forward with a prompt first!')
        batch_size = len(listbatch)
        # do everything on CPU to start, which I expect to be faster.
        input_ids = np.zeros((1,batch_size), dtype=np.int32)
        causal_mask = np.full((batch_size, batch_size), -10000)
        position_ids = np.zeros((1,batch_size), dtype=np.int32)
        # build map of where each parent is
        parent_map = { iids: i for i, iids in enumerate(listbatch) }
        for i, iids in enumerate(listbatch):
            input_ids[0,i] = iids[-1] # set the actual input id.
            position_ids[0,i] = seq_pos+len(iids)-1 # position in sequence
            for p in range(len(iids)-1):
                causal_mask[i,parent_map[iids[:p+1]]] = 0 # allow attention to parent.
            causal_mask[i,i] = 0 # allow attention to self.
        # now we can move these tensors to the GPU.
        input_ids = torch.tensor(input_ids, device=0, dtype=torch.long)
        causal_mask = torch.tensor(causal_mask, device=0, dtype=torch.bfloat16)
        position_ids = torch.tensor(position_ids, device=0, dtype=torch.long)
        return input_ids, position_ids, causal_mask

    def decode(self, batch_tree):
        input_ids, position_ids, causal_mask = self.build_decoder_batch(batch_tree)
        hidden_states = self.embeddings(input_ids, position_ids=position_ids)
        for layer in self.layers:
            hidden_states = layer.decode(hidden_states, causal_mask)
        return self.ln_f(hidden_states)

    def decode_listbatch(self, listbatch):
        input_ids, position_ids, causal_mask = self.build_decoder_listbatch(listbatch)
        hidden_states = self.embeddings(input_ids, position_ids=position_ids)
        for layer in self.layers:
            hidden_states = layer.decode(hidden_states, causal_mask)
        return self.ln_f(hidden_states)
    
    def update_kv(self, i):
        for layer in self.layers:
            layer.update_kv(i)
    
    def truncate_kv(self, pos):
        for layer in self.layers:
            layer.truncate_kv(pos)


class GPTLMHeadModel(GPTPreTrainedModel):

    def __init__(self, config: GPT2Config, device=None, dtype=None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(config)
        self.transformer = GPTModel(config, **kwargs, **factory_kwargs)
        pad_vocab_size_multiple = getattr(config, 'pad_vocab_size_multiple', 1)
        vocab_size = (math.ceil(config.vocab_size / pad_vocab_size_multiple)
                      * pad_vocab_size_multiple)
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False, **factory_kwargs)

    def forward(self, input_ids):
        """
            inference_params: for generation. Adapted from Megatron-LM (and Apex)
            https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        """
        hidden_states = self.transformer(input_ids)
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple('CausalLMOutput', ['logits'])
        return CausalLMOutput(logits=lm_logits)

    def decode(self, batch_tree):
        """
            inference_params: for generation. Adapted from Megatron-LM (and Apex)
            https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        """
        hidden_states = self.transformer.decode(batch_tree)
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple('CausalLMOutput', ['logits'])
        return CausalLMOutput(logits=lm_logits)

    def decode_listbatch(self, listbatch):
        """
            inference_params: for generation. Adapted from Megatron-LM (and Apex)
            https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        """
        hidden_states = self.transformer.decode_listbatch(listbatch)
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple('CausalLMOutput', ['logits'])
        return CausalLMOutput(logits=lm_logits)
    
    def update_kv(self, i):
        self.transformer.update_kv(i)
    
    def truncate_kv(self, pos):
        self.transformer.truncate_kv(pos)

    def load_state_dict(self, state_dict, strict=True):
        # Remapping from our checkpoints that used a different ordering of layers in the block
        # Previous: Attn / MLP -> Dropout -> Add -> LN
        # Current: Dropout -> Add -> LN -> Attn / MLP
        if 'transformer.ln_0.weight' in state_dict:
            n_layers = len(self.transformer.layers)
            ln_weight = state_dict.pop(f'transformer.layers.{n_layers - 1}.norm2.weight')
            ln_bias = state_dict.pop(f'transformer.layers.{n_layers - 1}.norm2.bias')
            state_dict['transformer.ln_f.weight'] = ln_weight
            state_dict['transformer.ln_f.bias'] = ln_bias
            for l in reversed(range(n_layers)):
                ln_weight = state_dict.pop(f'transformer.layers.{l}.norm1.weight')
                ln_bias = state_dict.pop(f'transformer.layers.{l}.norm1.bias')
                state_dict[f'transformer.layers.{l}.norm2.weight'] = ln_weight
                state_dict[f'transformer.layers.{l}.norm2.bias'] = ln_bias
                if l > 0:
                    ln_weight = state_dict.pop(f'transformer.layers.{l - 1}.norm2.weight')
                    ln_bias = state_dict.pop(f'transformer.layers.{l - 1}.norm2.bias')
                    state_dict[f'transformer.layers.{l}.norm1.weight'] = ln_weight
                    state_dict[f'transformer.layers.{l}.norm1.bias'] = ln_bias
            ln_weight = state_dict.pop('transformer.ln_0.weight')
            ln_bias = state_dict.pop('transformer.ln_0.bias')
            state_dict[f'transformer.layers.0.norm1.weight'] = ln_weight
            state_dict[f'transformer.layers.0.norm1.bias'] = ln_bias
        return super().load_state_dict(state_dict, strict=strict)


def remap_state_dict_gpt2(state_dict, config):
    # Word embedding and position embedding
    def key_mapping_pos_emb(key):
        return re.sub(r'^wpe.', 'transformer.embeddings.position_embeddings.', key)
    state_dict = OrderedDict((key_mapping_pos_emb(k), v) for k, v in state_dict.items())
    word_embeddings = state_dict.pop('wte.weight')
    # It's possible that vocab_size is padded to be a multiple of 8, for example.
    pad_vocab_size_multiple = getattr(config, 'pad_vocab_size_multiple', 1)
    vocab_size = (math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple)
    state_dict['transformer.embeddings.word_embeddings.weight'] = F.pad(
        word_embeddings, (0, 0, 0, vocab_size - word_embeddings.shape[0])
    )
    state_dict['lm_head.weight'] = state_dict['transformer.embeddings.word_embeddings.weight']

    # LayerNorm
    def key_mapping_ln(key):
        key = re.sub(r'^ln_f.(weight|bias)', r'transformer.ln_f.\1', key)
        key = re.sub(r'^h.(\d+).ln_(1|2).(weight|bias)', r'transformer.layers.\1.norm\2.\3', key)
        return key
    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLP
    for d in range(config.num_hidden_layers):
        W1 = state_dict.pop(f'h.{d}.mlp.c_fc.weight')
        state_dict[f'transformer.layers.{d}.mlp.fc1.weight'] = W1.t()
        W2 = state_dict.pop(f'h.{d}.mlp.c_proj.weight')
        state_dict[f'transformer.layers.{d}.mlp.fc2.weight'] = W2.t()
    def key_mapping_mlp(key):
        key = re.sub(r'^h.(\d+).mlp.c_fc.bias', r'transformer.layers.\1.mlp.fc1.bias', key)
        key = re.sub(r'^h.(\d+).mlp.c_proj.bias', r'transformer.layers.\1.mlp.fc2.bias', key)
        return key
    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # Attention
    for d in range(config.num_hidden_layers):
        state_dict.pop(f'h.{d}.attn.bias')  # We don't store this bias
        Wqkv = state_dict.pop(f'h.{d}.attn.c_attn.weight')
        state_dict[f'transformer.layers.{d}.mha.qkv.weight'] = Wqkv.t()
        Wout = state_dict.pop(f'h.{d}.attn.c_proj.weight')
        state_dict[f'transformer.layers.{d}.mha.out_proj.weight'] = Wout.t()
    def key_mapping_attn(key):
        key = re.sub(r'^h.(\d+).attn.c_attn.bias', r'transformer.layers.\1.mha.qkv.bias', key)
        key = re.sub(r'^h.(\d+).attn.c_proj.bias', r'transformer.layers.\1.mha.out_proj.bias', key)
        return key
    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())

    return state_dict