import torch

from models import GPTLMHeadModel

from util import *

def ngram_predict_tokens(ngrams, seq, num_return=999):
    if seq[-2:] in ngrams[2]:
        return ngrams[2][tuple(seq[-2:])][:num_return]
    elif seq[-1:] in ngrams[1]:
        return ngrams[1][(seq[-1],)][:num_return]
    else:
        return ngrams[0][:num_return]
    
def _generate_naive_batch(smallmodel: GPTLMHeadModel, accepted_tokens, seq_length, prev_2_tokens, batch_size=64, max_depth=None):
    # accepted_tokens are the tokens accepted by the big model
    # seq_length is the length taken up so far, not including accepted_tokens

    # the first element of the batch *must* be the last accepted token
    # which the big model has not yet seen autoregressively.

    # we need to reset the small model to be prepared to operate on the true accepted tokens.
    # print(f'Resetting to position {seq_length}')
    smallmodel.truncate_kv(seq_length)

    # we can now decode on all of those tokens at once
    # we first build the nested dict:
    accepted_tokens = tuple(accepted_tokens)
    initial_listbatch = []
    for i in range(len(accepted_tokens)):
        initial_listbatch.append(accepted_tokens[:i+1])
    # now we decode
    true_logits = smallmodel.decode_listbatch(initial_listbatch).logits[0,-1]
    smallmodel.update_kv(len(accepted_tokens)-1)
    truetoken = true_logits.argmax(dim=0).item() # decode greedily from all of them, all at once!
    # and then accept the results

    # our model is now in a position that we can start to generate a serious batch.

    # the first element of the batch, again, must be the last accepted token.
    # this is a token that the big model has *not* yet seen and yet is guaranteed.

    current_token = truetoken
    listbatch = [(prev_2_tokens[-1],), (prev_2_tokens[-1], truetoken)]

    counter = 0
    
    while len(listbatch) < batch_size:
        logits = smallmodel.decode_listbatch([(current_token,)]).logits[0,-1]
        smallmodel.update_kv(0)
        current_token = logits.argmax(dim=0).item()
        listbatch.append(listbatch[-1]+(current_token,))

        counter += 1

    origins = [1 for _ in listbatch]

    # print(listbatch)
    return listbatch, origins, counter

def _generate_staged_batch(smallmodel: GPTLMHeadModel, accepted_tokens, seq_length, prev_2_tokens, ngrams=None, batch_size=64, temperature=0.1, max_depth=None):
    # accepted_tokens are the tokens accepted by the big model
    # seq_length is the length taken up so far, not including accepted_tokens

    # the first element of the batch *must* be the last accepted token
    # which the big model has not yet seen autoregressively.

    # we need to reset the small model to be prepared to operate on the true accepted tokens.
    # print(f'Resetting to position {seq_length}')
    smallmodel.truncate_kv(seq_length)

    # we can now decode on all of those tokens at once
    # we first build the nested dict:
    accepted_tokens = tuple(accepted_tokens)
    initial_listbatch = []
    for i in range(len(accepted_tokens)):
        initial_listbatch.append(accepted_tokens[:i+1])
    # now we decode
    probs = torch.softmax(smallmodel.decode_listbatch(initial_listbatch).logits[0,-1] / temperature, dim=0)
    # and then accept the results
    smallmodel.update_kv(len(accepted_tokens)-1)

    # our model is now in a position that we can start to generate a serious batch.

    # the first element of the batch, again, must be the last accepted token.
    # this is a token that the big model has *not* yet seen and yet is guaranteed.

    # we should feel essentially no shame for doing lots of recomputation by never updating the kv cache here
    # because we get great parallelism with small models so it's not too painful.

    # overall algorithm:
    # at span=1, we assume the batch will be filled with the top batch_size tokens.
    # we then generate the following sequence and compute all joint probabilities.
    # we take the top batch_size and repeat
    # we stop when it doesn't change between iterations. (Technically stopping a tiny bit sooner is probably better but it doesn't make too much difference.)

    listbatch = []

    already_inserted = set()
    cumprob_cache = {(accepted_tokens[-1],): 1.0}

    min_prob = 0 # this is the minimum probability for a sequence to even be considered.
    
    # new plan for linear generation
    # we use the ngram model to generate a batch for the speculative model and predict forward
    # and we always just cache the logits

    # then, once we've reached appropriate depth, we do our standard packing of the batch with the most likely cases.

    # one assumption is that updating KV is not worth it here because it's too expensive to be worth it. Unsure if true or not.

    lvl_1_tokens = 12
    lvl_2_tokens = 7
    lvl_3_tokens = 4
    lvl_4_tokens = 1
    lvl_5_tokens = 1
    lvl_6_tokens = 1

    fullseq = prev_2_tokens[:] # copy
    fullseq_origins = [0,0] # first token always originated with the big model

    counter = 0
    while len(fullseq) < max_depth:
        # print('\n\nSTARTING ITERATION', counter, '\n\n')

        listbatch = [fullseq[1:i+1] for i in range(1, len(fullseq))]
        # print('ORIGINAL LISTBATCH:')
        # pprint(listbatch)
        # now we need to extend it with the most likely tokens
        lvl_1_predictions = ngram_predict_tokens(ngrams, listbatch[-1], num_return=lvl_1_tokens)
        listbatch.extend([listbatch[-1]+(x,) for x in lvl_1_predictions])
        lvl_1_offset = len(lvl_1_predictions)
        # now we do the following tokens, too!
        lvl_2_predictions = ngram_predict_tokens(ngrams, listbatch[-lvl_1_offset], num_return=lvl_2_tokens)
        listbatch.extend([listbatch[-lvl_1_offset]+(x,) for x in lvl_2_predictions]) # this is the location of the most likely token!
        lvl_2_offset = len(lvl_2_predictions)
        # and again for level 3
        lvl_3_predictions = ngram_predict_tokens(ngrams, listbatch[-lvl_2_offset], num_return=lvl_3_tokens)
        listbatch.extend([listbatch[-lvl_2_offset]+(x,) for x in lvl_3_predictions])
        lvl_3_offset = len(lvl_3_predictions)
        # and again for level 4
        lvl_4_predictions = ngram_predict_tokens(ngrams, listbatch[-lvl_3_offset], num_return=lvl_4_tokens)
        listbatch.extend([listbatch[-lvl_3_offset]+(x,) for x in lvl_4_predictions])
        lvl_4_offset = len(lvl_4_predictions)
        # and again for level 5
        lvl_5_predictions = ngram_predict_tokens(ngrams, listbatch[-lvl_4_offset], num_return=lvl_5_tokens)
        listbatch.extend([listbatch[-lvl_4_offset]+(x,) for x in lvl_5_predictions])
        lvl_5_offset = len(lvl_5_predictions)
        # and again for level 6
        lvl_6_predictions = ngram_predict_tokens(ngrams, listbatch[-lvl_5_offset], num_return=lvl_6_tokens)
        listbatch.extend([listbatch[-lvl_5_offset]+(x,) for x in lvl_6_predictions])
        # ^ the above could easily be factored into a loop but I actually think it would make it less readable. So, I'm not doing it until I figure out how to make it intuitive.

        # now we run the batch!
        probs = torch.softmax(smallmodel.decode_listbatch(listbatch).logits[0] / temperature, dim=1)
        topk_val, topk_idx = torch.topk(probs.type(torch.float32), 4, dim=1)
        topk_idx = topk_idx.tolist()
        topk_val = topk_val.tolist()
        # now we need to update the cache, which will determine our final listbatch at the end.
        for i, (k, idxs, vals) in enumerate(zip(listbatch, topk_idx, topk_val)):
            if k not in already_inserted and k in cumprob_cache:
                already_inserted.add(k)
                for idx, val in zip(idxs, vals):
                    cumval = val*cumprob_cache[k]
                    if cumval < min_prob:
                        break # no others will be useful
                    cumprob_cache[k+(idx,)] = cumval

        seq_to_pos = { k: i for i, k in enumerate(listbatch) }

        # update min_prob to prevent cumprob_cache from growing too large
        cumprob_values = list(cumprob_cache.values())
        min_prob = sorted(cumprob_values, reverse=True)[min(len(cumprob_values), batch_size)-1]
                
        truetokens = probs.argmax(dim=1).tolist() # decode greedily from all of them, all at once!

        token_tuple = tuple(fullseq[1:])
        i = seq_to_pos[token_tuple]
        while True:
            truetoken = truetokens[i]
            fullseq += (truetoken,)
            token_tuple = tuple(fullseq[1:])
            if token_tuple in seq_to_pos:
                fullseq_origins.append(3) # originally came from ngram
                i = seq_to_pos[token_tuple]
            else:
                fullseq_origins.append(1) # originally came from small model
                # we're shit outta luck!
                break

        counter += 1 # we did run another batch, after all. need to report that!

    sorted_logprob_cache = sorted(list(cumprob_cache.items()), key=lambda x: x[1], reverse=True)

    tup_to_origin = { tuple(fullseq[1:k+1]): v for k, v in zip(range(len(fullseq)), fullseq_origins) }
    # pprint(tup_to_origin)

    # there may actually be a slight bug here in the case of an actual tie. But I'm not sure if it's worth fixing.
    listbatch = [x[0] for x in sorted_logprob_cache[:batch_size]] # I'm pretty sure this has to respect causal orderings

    origins = [tup_to_origin.get(x, 1) for x in listbatch]
    
    return listbatch, origins, counter

def generate_speculative(batch_gen_fn, input_ids, smallmodel: GPTLMHeadModel, bigmodel: GPTLMHeadModel, ntokens=100, topk=50, temperature=1.0):
    with torch.inference_mode():
        # preprocess with both models
        _ = smallmodel(input_ids).logits[0,-1]
        first_logits = bigmodel(input_ids).logits[0,-1]
        truetoken = first_logits.argmax().item()
        # print(f'first token: "{tokenizer.decode(truetoken)}"')
        # first_logits -= torch.max(first_logits) in case of sampling above a threshhold
        # generate batch_size tokens using the small model.
        output_tokens = [truetoken]
        accepted_tokens = [truetoken]
        nbatch = 0 # count how many batches we need
        small_nbatch = 0 # count how many batches we need for the small model
        origins = [-1]*input_ids.shape[1] + [0] # the model level in {-1, 0, 1, 2} where the token originated.
        while len(output_tokens) < ntokens:
            nbatch += 1
            seqpos_to_truncate = input_ids.shape[1]+len(output_tokens)-len(accepted_tokens)
            # print('truncating to ', seqpos_to_truncate, 'tokens')
            prev_2_tokens = tuple(output_tokens[-2:])
            if len(prev_2_tokens) == 1:
                prev_2_tokens = (input_ids[0][-1].item(), prev_2_tokens[0])
            seqbatch, originsbatch, s_nbatch = batch_gen_fn(smallmodel, accepted_tokens, seqpos_to_truncate, prev_2_tokens)
            small_nbatch += s_nbatch
            seq_to_pos = { seq: i for i, seq in enumerate(seqbatch) }
            # now feed this batch into the big model
            true_logits = bigmodel.decode_listbatch(seqbatch).logits[0]
            if temperature > 0:
                # sample truetokens from true_logits using topk and softmax with temperature
                topk_logits = torch.topk(true_logits, topk, dim=-1)
                # sample tokens from true_probs
                trueindices = torch.multinomial(torch.softmax(topk_logits.values/temperature, dim=-1), 1).squeeze(-1).tolist()
                truetokens = []
                for i in range(len(trueindices)):
                    truetokens.append(topk_logits.indices[i][trueindices[i]].item())
            else:
                truetokens = true_logits.argmax(dim=1).tolist() # decode greedily from all of them, all at once!
            # print(true_logits.shape)
            # we now accept as many as we can until we turn out to be wrong.
            accepted_tokens = [seqbatch[0][0]]
            accepted_origins = [0]
            token_tuple = tuple(accepted_tokens)
            i = seq_to_pos[token_tuple]
            while True:
                truetoken = truetokens[i] # this is annoyingly expensive to get off of GPU but I don't have a better way.
                accepted_tokens.append(truetoken)
                token_tuple = tuple(accepted_tokens)
                if token_tuple in seq_to_pos:
                    i = seq_to_pos[token_tuple]
                    accepted_origins.append(originsbatch[i])
                else:
                    # we're shit outta luck!
                    accepted_origins.append(0)
                    break
            # update kv cache
            bigmodel.update_kv(seq_to_pos[tuple(accepted_tokens[:-1])]) # update on the position of everything except whatever we just failed on

            output_tokens.extend(accepted_tokens[1:])
            origins.extend(accepted_origins[1:])
        print(f'Used {nbatch} big batches and {small_nbatch} small batches to generate {len(output_tokens)} tokens.')
        oids = torch.tensor(input_ids.reshape((-1,)).tolist()+output_tokens,dtype=torch.long).reshape((1,-1))
        return oids, origins