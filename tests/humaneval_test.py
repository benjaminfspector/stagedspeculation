import re

import torch

import pickle, json

from transformers import GPT2Config
from transformers import AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel as GPT2LMHeadModelHF

import cProfile

import sys
# add directory up as part of path
sys.path.append('../src')

from tqdm import tqdm
from models import GPTLMHeadModel, remap_state_dict_gpt2, state_dict_from_pretrained

from util import *
from speculative import generate_speculative, _generate_staged_batch, _generate_naive_batch

from functools import partial

import datetime

from termcolor import cprint, colored
from pprint import pprint

NTOKENS = 250
NITER_WARMUP = 0
NITER_TEST = 1

PROFILING = False

MODELSDIR = '../models/'
bigmodelname = 'gpt-750M-pythonstack'
smallmodelname = 'gpt-40M-pythonstack'

# ------------- END CONFIG ----------------

with open('data/HumanEval.jsonl', 'r') as f:
    prompts = [json.loads(x)["prompt"] for x in f.readlines()]
print('Number of prompts:', len(prompts))

print(f'Model: {bigmodelname}')
print(f'Draft model: {smallmodelname}')

bigmodel_path = f'{MODELSDIR}/{bigmodelname}'
bigconfig = GPT2Config.from_pretrained(bigmodel_path)
bigmodel = GPTLMHeadModel.from_pretrained(bigmodel_path, bigconfig, dtype=torch.bfloat16).to(0)
bigmodel.eval()

smallmodel_path = f'{MODELSDIR}/{smallmodelname}'
smallconfig = GPT2Config.from_pretrained(smallmodel_path)
smallmodel = GPTLMHeadModel.from_pretrained(smallmodel_path, smallconfig, dtype=torch.bfloat16).to(0)
smallmodel.eval()

# refmodel = GPT2LMHeadModelHF.from_pretrained(bigmodel_path).to(0)

print('Finished loading models!')

with open('../models/ngram_nofreq.pkl', 'rb') as f:
    ngrams = pickle.load(f)
    ngrams = [ngrams['unigrams'], ngrams['bigrams'], ngrams['trigrams']]
    print('unigrams:', len(ngrams[0]))
    print('bigrams:', len(ngrams[1]))
    print('trigrams:', len(ngrams[2]))

print('Finished loading ngrams!')

tokenizer = AutoTokenizer.from_pretrained('gpt2')

tokens = [tokenizer.encode(prompt, return_tensors='pt').to(0) for prompt in prompts][:1]
print('INPUT SHAPE', [t.shape for t in tokens])

coloring_args = {
    -2: ['green', 'on_white'], # reference
    -1: ['white', 'on_dark_grey'], # prompt
    0: ['white', 'on_red'], # main model
    1: ['white', 'on_blue'], # small (draft) model
    2: ['light_yellow', 'on_black'], # tiny model
    3: ['white', 'on_green'], # ngram (draft^2) model
}

stagedspec_elapsedtimes = []
stagedspec_batch_fn = partial(_generate_staged_batch, ngrams=ngrams, batch_size=36, max_depth=8, temperature=1.0)
if PROFILING:
    with cProfile.Profile() as pr:
        for i, tok in tqdm(enumerate(tokens)):
            # generate speculative output
            resp, elapsedtime = timefn(stagedspec_batch_fn, f'STAGED SPECULATIVE {i+1}', NITER_WARMUP, NITER_TEST, stagedspec_batch_fn, tok, smallmodel, bigmodel, ntokens=NTOKENS, temperature=0.0)
            all_tokens, origins = resp
            tokens_string = tokenizer.batch_decode(all_tokens.reshape((-1,1)), skip_special_tokens=True)
            colored_tokens_string = ''.join([colored(x, *coloring_args[y]) for x,y in zip(tokens_string, origins)])
            print(colored_tokens_string)
            print(f'\nElapsed time for {NTOKENS} tokens: {elapsedtime} s | {elapsedtime/NITER_TEST} s/it.')
            printheader(f'ENDING STAGED SPECULATIVE {i+1}')
            stagedspec_elapsedtimes.append(elapsedtime)
        pr.print_stats()
        pr.dump_stats(f"spec_{str(datetime.datetime.now()).replace(' ', '_')}.prof")
else:
    for i, tok in tqdm(enumerate(tokens)):
        resp, elapsedtime = timefn(generate_speculative, f'STAGED SPECULATIVE {i+1}', NITER_WARMUP, NITER_TEST, stagedspec_batch_fn, tok, smallmodel, bigmodel, ntokens=NTOKENS, temperature=0.0)
        all_tokens, origins = resp
        tokens_string = tokenizer.batch_decode(all_tokens.reshape((-1,1)), skip_special_tokens=True)
        colored_tokens_string = ''.join([colored(x, *coloring_args[y]) for x,y in zip(tokens_string, origins)])
        print(colored_tokens_string)
        print(f'\nElapsed time for {NTOKENS} tokens: {elapsedtime} s | {elapsedtime/NITER_TEST} s/it.')
        printheader(f'ENDING STAGED SPECULATIVE {i+1}')
        stagedspec_elapsedtimes.append(elapsedtime)
print('Staged speculative time per prompt:', [s / NITER_TEST for s in stagedspec_elapsedtimes])
with open('humaneval_stagedspec_times.json', 'w') as f:
    json.dump(stagedspec_elapsedtimes, f, indent=2)

naivespec_elapsedtimes = []
naivespec_batch_fn = partial(_generate_naive_batch, batch_size=6)
if PROFILING:
    with cProfile.Profile() as pr:
        for i, tok in tqdm(enumerate(tokens)):
            # generate speculative output
            resp, elapsedtime = timefn(generate_speculative, f'NAIVE SPECULATIVE {i+1}', NITER_WARMUP, NITER_TEST, naivespec_batch_fn, tok, smallmodel, bigmodel, ntokens=NTOKENS, temperature=0.0)
            all_tokens, origins = resp
            tokens_string = tokenizer.batch_decode(all_tokens.reshape((-1,1)), skip_special_tokens=True)
            colored_tokens_string = ''.join([colored(x, *coloring_args[y]) for x,y in zip(tokens_string, origins)])
            print(colored_tokens_string)
            print(f'\nElapsed time for {NTOKENS} tokens: {elapsedtime} s | {elapsedtime/NITER_TEST} s/it.')
            printheader(f'ENDING NAIVE SPECULATIVE {i+1}')
            naivespec_elapsedtimes.append(elapsedtime)
        pr.print_stats()
        pr.dump_stats(f"spec_{str(datetime.datetime.now()).replace(' ', '_')}.prof")
else:
    for i, tok in tqdm(enumerate(tokens)):
        resp, elapsedtime = timefn(generate_speculative, f'NAIVE SPECULATIVE {i+1}', NITER_WARMUP, NITER_TEST, naivespec_batch_fn, tok, smallmodel, bigmodel, ntokens=NTOKENS, temperature=0.0)
        all_tokens, origins = resp
        tokens_string = tokenizer.batch_decode(all_tokens.reshape((-1,1)), skip_special_tokens=True)
        colored_tokens_string = ''.join([colored(x, *coloring_args[y]) for x,y in zip(tokens_string, origins)])
        print(colored_tokens_string)
        print(f'\nElapsed time for {NTOKENS} tokens: {elapsedtime} s | {elapsedtime/NITER_TEST} s/it.')
        printheader(f'ENDING NAIVE SPECULATIVE {i+1}')
        naivespec_elapsedtimes.append(elapsedtime)
print('Naive speculative time per prompt:', [s / NITER_TEST for s in naivespec_elapsedtimes])
with open('humaneval_naivespec_times.json', 'w') as f:
    json.dump(naivespec_elapsedtimes, f, indent=2)

def generate_model_naive(model, tokens, ntokens, batch_size):
    all_tokens = list(tokens.reshape(-1))
    first_logits = model(tokens).logits[0,-1]
    besttoken = first_logits.argmax().item()
    all_tokens.append(besttoken)
    for _ in range(ntokens-1):
        mybatch = {
            besttoken+i: {} for i in range(batch_size)
        }
        besttoken = model.decode(mybatch).logits[0,0].argmax().item()
        all_tokens.append(besttoken)
        model.update_kv(0)
    torch.cuda.synchronize()
    return torch.tensor(all_tokens, dtype=torch.long).reshape((1,-1))

b1_elapsedtimes = []
with torch.inference_mode():
    for i, tok in tqdm(enumerate(tokens)):
        all_tokens, elapsedtime = timefn(generate_model_naive, f'LARGE MODEL {i+1} BATCH=1', NITER_WARMUP, NITER_TEST, bigmodel, tok, NTOKENS, 1)
        tokens_string = tokenizer.batch_decode(all_tokens.reshape((-1,1)), skip_special_tokens=True)
        origins = [-1]*tok.shape[-1] + [0]*(NTOKENS+20)
        colored_tokens_string = ''.join([colored(x, *coloring_args[y]) for x,y in zip(tokens_string, origins)])
        print(colored_tokens_string)
        print(f'\nElapsed time for {NTOKENS} tokens: {elapsedtime} s | {elapsedtime/NITER_TEST} s/it.')
        printheader(f'ENDING LARGE MODEL {i+1} BATCH=1')
        b1_elapsedtimes.append(elapsedtime)
print('Normal time per prompt:', [s / NITER_TEST for s in b1_elapsedtimes])
with open('humaneval_b1_times.json', 'w') as f:
    json.dump(b1_elapsedtimes, f, indent=2)

# ref_elapsedtimes = []
# for i, tok in enumerate(tokens):
#     # generate reference output
#     x, elapsedtime = timefn(refmodel.generate, f'REFERENCE OUTPUT {i+1}', NITER_WARMUP, NITER_TEST, tok, max_new_tokens=NTOKENS)
#     tokens_string = tokenizer.batch_decode(x.reshape((-1,1)), skip_special_tokens=True)
#     origins = [-1]*tok.shape[-1] + [-2]*(NTOKENS+20)
#     colored_tokens_string = ''.join([colored(x, *coloring_args[y]) for x,y in zip(tokens_string, origins)])
#     print(colored_tokens_string)
#     print(f'\nElapsed time for {NTOKENS} tokens: {elapsedtime} s | {elapsedtime/NITER_TEST} s/it.')
#     printheader(f'ENDING REFERENCE OUTPUT {i+1}')
#     ref_elapsedtimes.append(elapsedtime)
# print('Reference time per prompt:', [s / NITER_TEST for s in ref_elapsedtimes])

# with torch.inference_mode():

#     b1_elapsedtimes = []
#     for i, tok in enumerate(tokens):
#         all_tokens, elapsedtime = timefn(generate_model_naive, f'LARGE MODEL {i+1} BATCH=1', NITER_WARMUP, NITER_TEST, bigmodel, tok, NTOKENS, 1)
#         tokens_string = tokenizer.batch_decode(all_tokens.reshape((-1,1)), skip_special_tokens=True)
#         origins = [-1]*tok.shape[-1] + [0]*(NTOKENS+20)
#         colored_tokens_string = ''.join([colored(x, *coloring_args[y]) for x,y in zip(tokens_string, origins)])
#         print(colored_tokens_string)
#         print(f'\nElapsed time for {NTOKENS} tokens: {elapsedtime} s | {elapsedtime/NITER_TEST} s/it.')
#         printheader(f'ENDING LARGE MODEL {i+1} BATCH=1')
#         b1_elapsedtimes.append(elapsedtime)

    # for i, tok in enumerate(tokens):
    #     all_tokens, elapsedtime = timefn(generate_model_naive, f'SMALL MODEL {i+1} BATCH=1', NITER_WARMUP, NITER_TEST, smallmodel, tok, NTOKENS, 1)
    #     tokens_string = tokenizer.batch_decode(all_tokens.reshape((-1,1)), skip_special_tokens=True)
    #     origins = [-1]*tok.shape[-1] + [1]*(NTOKENS+20)
    #     colored_tokens_string = ''.join([colored(x, *coloring_args[y]) for x,y in zip(tokens_string, origins)])
    #     print(colored_tokens_string)
    #     print(f'\nElapsed time for {NTOKENS} tokens: {elapsedtime} s | {elapsedtime/NITER_TEST} s/it.')
    #     printheader(f'ENDING SMALL MODEL {i+1} BATCH=1')

    # for i, tok in enumerate(tokens):
    #     all_tokens, elapsedtime = timefn(generate_model_naive, f'TINY MODEL {i+1} BATCH=1', NITER_WARMUP, NITER_TEST, tinymodel, tok, NTOKENS, 1)
    #     tokens_string = tokenizer.batch_decode(all_tokens.reshape((-1,1)), skip_special_tokens=True)
    #     origins = [-1]*tok.shape[-1] + [2]*(NTOKENS+20)
    #     colored_tokens_string = ''.join([colored(x, *coloring_args[y]) for x,y in zip(tokens_string, origins)])
    #     print(colored_tokens_string)
    #     print(f'\nElapsed time for {NTOKENS} tokens: {elapsedtime} s | {elapsedtime/NITER_TEST} s/it.')
    #     printheader(f'ENDING TINY MODEL {i+1} BATCH=1')

# print('Batch=1 time per prompt:', [s / NITER_TEST for s in b1_elapsedtimes])
# print('Speculative time per prompt:', [s / NITER_TEST for s in spec_elapsedtimes])
# print('Reference time per prompt:', [s / NITER_TEST for s in ref_elapsedtimes])