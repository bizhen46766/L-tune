import torch

from utils import get_verbalization_ids


class PromptEncoder(object):
    def __init__(self, tokenizer, pvp, label_list):
        # TODO init embedding from different tokens??????
        # Because RoBERTa's tokenizer tokenize same words differently in different positions
        # (beginning of sentence / inside a sentence),
        # here we record the index of unique tokens to map tokens to embeddings
        pattern_token_to_index, pattern_token_indices = {}, []
        for idx, part in enumerate(pvp.PATTERN):
            if pvp.BLOCK_FLAG[idx] == 1:
                for token in tokenizer.convert_tokens_to_ids(tokenizer.tokenize(part)):
                    if token not in pattern_token_to_index:
                        pattern_token_to_index[token] = len(
                            pattern_token_to_index)
                pattern_token_indices.append(pattern_token_to_index[token])

        label_token_set = set()
        for label_idx, label in enumerate(label_list):
            verbalizers = pvp.verbalize(label)
            for verbalizer_idx, verbalizer in enumerate(verbalizers):
                verbalizer_id = get_verbalization_ids(
                    verbalizer, tokenizer, force_single_token=True)
                assert verbalizer_id != tokenizer.unk_token_id, "verbalization was tokenized as <UNK>"
                label_token_set.add(verbalizer_id)

        assert len(pattern_token_to_index) < 50 and len(label_token_set) < 49

        # Convert tokens in manual prompt / label to unused tokens
        # Note that `AlbertTokenizer` doesn't have a `vocab` attribute
        if hasattr(tokenizer, 'vocab') and '[unused0]' in tokenizer.vocab:
            # BERT
            self.pattern_convert = {token_id: tokenizer.vocab['[unused%s]' % idx]
                                    for idx, token_id in enumerate(pattern_token_set)}
            self.label_convert = {token_id: tokenizer.vocab['[unused%s]' % (idx + 50)]
                                  for idx, token_id in enumerate(label_token_set)}

        else:
            # ALBERT, RoBERTa
            start_idx = tokenizer.vocab_size - 100
            self.pattern_convert = {token_id: start_idx + idx
                                    for idx, token_id in enumerate(pattern_token_set)}
            self.label_convert = {token_id: start_idx + 50 + idx
                                  for idx, token_id in enumerate(label_token_set)}

        # Convert mlm logits to cls logits
        self.m2c_tensor = torch.tensor(
            list(self.label_convert.values()), dtype=torch.long)

    def init_embed(self, model):
        w = model.get_input_embeddings().weight.data
        for origin_id, convert_id in self.pattern_convert.items():
            w[convert_id] = w[origin_id]
        for origin_id, convert_id in self.label_convert.items():
            w[convert_id] = w[origin_id]

    def add_embed_hook(self, model):
        def stop_gradient(_, grad_input, __):
            # grad_input: tuple containing a (vocab_size, hidden_dim) tensor
            # grad_output: tuple containing a (batch_size, max_seq_len, hidden_dim) tensor
            return (grad_mask.to(grad_input[0].device) * grad_input[0],)

        # Train certain tokens by multiply gradients with a mask
        trainable_ids = list(self.pattern_convert.values()) + \
            list(self.label_convert.values())
        grad_mask = torch.zeros((len(tokenizer.vocab), 1), dtype=torch.float)
        grad_mask[trainable_ids, 0] = 1.0

        return model.get_input_embeddings().register_backward_hook(stop_gradient)

    def convert_input_ids(self, input_ids, block_flag):
        bz = len(input_ids)
        for bidx in range(bz):
            indices = torch.where(block_flag[bidx] == 1)
            for idx in indices:
                input_ids[bidx][idx] = self.pattern_convert[input_ids[bidx][idx].item()]

        return input_ids

    def get_replace_embeds(self, input_ids, block_flag, word_embeddings):
        # Use first sample's block flag
        indices = torch.where(block_flag[0] == 1)[0]
        convert_ids = [self.pattern_convert[input_ids[0][idx].item()]
                       for idx in indices]
        lookup_tensor = torch.tensor(
            convert_ids, dtype=torch.long).to(input_ids.device)

        return word_embeddings(lookup_tensor)

    def convert_mlm_logits_to_cls_logits(self, mlm_labels, logits):
        return torch.index_select(logits[mlm_labels != -1], -1, self.m2c_tensor.to(logits.device))
