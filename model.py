# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains code for wrapping a transformer language model and
provides convenience methods for training and inference.
"""

import json
import jsonpickle
import os
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import trange, tqdm
from transformers import InputExample, AdamW, get_linear_schedule_with_warmup, \
    BertForMaskedLM, RobertaForMaskedLM, BertConfig, BertTokenizer, RobertaConfig, \
    RobertaTokenizer, AlbertForMaskedLM, AlbertTokenizer, AlbertConfig

import logging
from data_utils import PVPS, load_task_helper, evaluate_results
from config import WrapperConfig, EvalConfig
from utils import InputFeatures, DictDataset
from encoder import PromptEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('model')

CONFIG_NAME = 'wrapper_config.json'

MODEL_CLASSES = {
    'bert': {'config': BertConfig, 'tokenizer': BertTokenizer, 'model': BertForMaskedLM},
    'roberta': {'config': RobertaConfig, 'tokenizer': RobertaTokenizer, 'model': RobertaForMaskedLM},
    'albert': {'config': AlbertConfig, 'tokenizer': AlbertTokenizer, 'model': AlbertForMaskedLM}
}


class ContinuousPrompt(nn.Module):
    def __init__(self, config: WrapperConfig, tokenizer, pvp):
        super(ContinuousPrompt, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.embed_size = config.embed_size
        self.hidden_size = self.embed_size

        # The pattern_id is supposed to indicate the number of continuous prompt tokens.
        prompt_length = 0
        for idx, val in enumerate(pvp.BLOCK_FLAG):
            if val == 1:
                prompt_length += len(tokenizer.tokenize(pvp.PATTERN[idx]))
        self.prompt_length = prompt_length

        config_class = MODEL_CLASSES[self.config.model_type]['config']
        model_config = config_class.from_pretrained(
            config.model_name_or_path,
            num_labels=len(config.label_list),
            finetuning_task=config.task_name,
            cache_dir=config.cache_dir if config.cache_dir else None,
            use_cache=False)

        model_class = MODEL_CLASSES[self.config.model_type]['model']
        self.model = model_class.from_pretrained(
            config.model_name_or_path,
            config=model_config,
            cache_dir=config.cache_dir if config.cache_dir else None)

        self.prompt_embeddings = torch.nn.Embedding(
            self.prompt_length, self.embed_size)
        if config.prompt_encoder_type == "lstm":
            self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                           hidden_size=self.hidden_size,
                                           num_layers=2,
                                           bidirectional=True,
                                           batch_first=True)
            self.mlp_head = nn.Sequential(nn.Linear(2 * self.hidden_size, self.hidden_size),
                                          nn.ReLU(),
                                          nn.Linear(self.hidden_size, self.hidden_size))

        elif config.prompt_encoder_type == "mlp":
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_size, self.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_size, self.hidden_size))

        elif config.prompt_encoder_type in {"none", "inner"}:
            # Manual prompt without continuous tuning, or:
            # Use some unused tokens as prompt tokens / label tokens
            pass

        else:
            raise ValueError('unknown prompt_encoder_type.')


class TransformerModelWrapper(object):
    """A wrapper around a Transformer-based language model."""

    def __init__(self, config: WrapperConfig):
        self.config = config

        tokenizer_class = MODEL_CLASSES[config.model_type]['tokenizer']
        self.tokenizer = tokenizer_class.from_pretrained(
            config.model_name_or_path,
            cache_dir=config.cache_dir if config.cache_dir else None)

        self.pvp = PVPS[config.task_name](self, config.pattern_id)
        self.model = ContinuousPrompt(config, self.tokenizer, self.pvp)
        self.task_helper = load_task_helper(config.task_name, self)
        self.label_map = {label: i for i,
                          label in enumerate(self.config.label_list)}

        if config.prompt_encoder_type == "inner":
            self.encoder = PromptEncoder(
                self.tokenizer, self.pvp, config.label_list)
            self.encoder.init_embed(self.model.model)

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.cuda()

    def save(self, path: str) -> None:
        logger.info("Saving models.")
        model_to_save = self.model.module if hasattr(
            self.model, 'module') else self.model

        model_to_save.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self._save_config(path)

        if self.config.prompt_encoder_type == "lstm":
            state = {
                "prompt_embeddings": model_to_save.prompt_embeddings.state_dict(),
                "lstm_head": model_to_save.lstm_head.state_dict(),
                "mlp_head": model_to_save.mlp_head.state_dict()
            }
        elif self.config.prompt_encoder_type == "mlp":
            state = {
                "prompt_embeddings": model_to_save.prompt_embeddings.state_dict(),
                "mlp": model_to_save.mlp.state_dict()
            }
        elif self.config.prompt_encoder_type in {"none", "inner"}:
            state = {}
        else:
            raise ValueError("unknown prompt_encoder_type.")

        save_path_file = os.path.join(path, "embeddings.pth")
        torch.save(state, save_path_file)

    @classmethod
    def from_pretrained(cls, path: str) -> 'TransformerModelWrapper':
        """Load a pretrained wrapper from a given path."""

        wrapper = TransformerModelWrapper.__new__(TransformerModelWrapper)
        wrapper.config = wrapper._load_config(path)

        tokenizer_class = MODEL_CLASSES[wrapper.config.model_type]['tokenizer']
        wrapper.tokenizer = tokenizer_class.from_pretrained(path)

        wrapper.pvp = PVPS[wrapper.config.task_name](
            wrapper, wrapper.config.pattern_id)

        wrapper.model = ContinuousPrompt(
            wrapper.config, wrapper.tokenizer, wrapper.pvp)
        model_class = MODEL_CLASSES[wrapper.config.model_type]['model']
        wrapper.model.model = model_class.from_pretrained(path)

        # Load prompt embeddings
        save_path_file = os.path.join(path, "embeddings.pth")
        data = torch.load(save_path_file)

        # `inner` / `none` encoder
        if "prompt_embeddings" in data:
            wrapper.model.prompt_embeddings.load_state_dict(
                data["prompt_embeddings"])

        if "lstm_head" in data:
            assert ("mlp_head" in data)
            wrapper.model.lstm_head.load_state_dict(data["lstm_head"])
            wrapper.model.mlp_head.load_state_dict(data["mlp_head"])
        if "mlp" in data:
            wrapper.model.mlp_head.load_state_dict(data["mlp"])

        if wrapper.config.prompt_encoder_type == "inner":
            wrapper.encoder = PromptEncoder(
                wrapper.tokenizer, wrapper.pvp, wrapper.config.label_list)

        wrapper.label_map = {label: i for i,
                             label in enumerate(wrapper.config.label_list)}
        wrapper.task_helper = load_task_helper(
            wrapper.config.task_name, wrapper)

        if torch.cuda.device_count() > 1:
            wrapper.model = torch.nn.DataParallel(wrapper.model)
        wrapper.model.cuda()

        return wrapper

    def _save_config(self, path: str) -> None:
        with open(os.path.join(path, CONFIG_NAME), 'w') as f:
            f.write(jsonpickle.encode(self.config))

    @staticmethod
    def _load_config(path: str) -> WrapperConfig:
        with open(os.path.join(path, CONFIG_NAME), 'r') as f:
            return jsonpickle.decode(f.read())

    def train(self,
              train_data: List[InputExample],
              eval_data: List[InputExample],
              dev_data: List[InputExample],
              eval_config: EvalConfig,
              pattern_iter_output_dir,
              per_gpu_train_batch_size: int = 8,
              n_gpu: int = 1,
              num_train_epochs: int = 3,
              gradient_accumulation_steps: int = 1,
              weight_decay: float = 0.0,
              learning_rate: float = 5e-5,
              adam_epsilon: float = 1e-8,
              warmup_steps=0,
              max_grad_norm: float = 1,
              logging_steps: int = 50,
              max_steps=-1, **kwargs):
        """
        Train the underlying language model.

        :param train_data: the training examples to use
        :param per_gpu_train_batch_size: the number of training examples per batch and gpu
        :param n_gpu: the number of gpus to use
        :param num_train_epochs: the number of epochs to train
        :param gradient_accumulation_steps: the number of gradient accumulation steps before performing an update
        :param weight_decay: the weight decay to use
        :param learning_rate: the learning rate to use
        :param adam_epsilon: epsilon parameter for the Adam optimizer
        :param warmup_steps: the number of warmup steps
        :param max_grad_norm: the maximum norm for the gradient
        :param logging_steps: the number of steps after which logging information is printed
        :param max_steps: the maximum number of training steps, overrides ``num_train_epochs``
        :return: a tuple consisting of the total number of steps and the average training loss
        """

        train_batch_size = per_gpu_train_batch_size * max(1, n_gpu)
        train_dataset = self._generate_dataset(train_data)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=train_batch_size)

        if max_steps > 0:
            t_total = max_steps
            num_train_epochs = max_steps // (
                max(1, len(train_dataloader) // gradient_accumulation_steps)) + 1
        else:
            t_total = len(
                train_dataloader) // gradient_accumulation_steps * num_train_epochs

        print("\nnum_steps_per_dataset:", len(
            train_dataloader) // gradient_accumulation_steps)
        print("total_steps:", t_total)
        print("num_train_epochs:", num_train_epochs)

        cur_model = self.model.module if hasattr(
            self.model, 'module') else self.model

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in cur_model.model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in cur_model.model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        embedding_parameters = None

        if self.config.prompt_encoder_type == "lstm":
            embedding_parameters = [
                {'params': [p for p in cur_model.lstm_head.parameters()]},
                {'params': [p for p in cur_model.mlp_head.parameters()]},
                {'params': [p for p in cur_model.prompt_embeddings.parameters()]}
            ]
        elif self.config.prompt_encoder_type == "mlp":
            embedding_parameters = [
                {'params': [p for p in cur_model.mlp.parameters()]},
                {'params': [p for p in cur_model.prompt_embeddings.parameters()]}
            ]
        elif self.config.prompt_encoder_type == "none":
            pass
        elif self.config.prompt_encoder_type == "inner":
            stage = kwargs.get('stage', 0)
            if stage == 1:
                # Training stage 1: only optimize prompt-related tokens
                handle = self.encoder.add_embed_hook(self.model.model)
                optimizer_grouped_parameters = [
                    {'params': [p for p in cur_model.model.get_input_embeddings().parameters()],
                     'weight_decay': weight_decay}]
            elif stage == 2:
                # Training stage 2: optimize all model weights
                pass
                # optimizer_grouped_parameters[0] = {'params': [p for n, p in cur_model.model.named_parameters(
                # ) if not any(nd in n for nd in no_decay + ['word_embeddings'])], 'weight_decay': weight_decay}
            else:
                # Normal training
                # embedding_parameters = [
                #     {'params': [
                #         p for p in cur_model.model.get_input_embeddings().parameters()]}
                # ]
                # optimizer_grouped_parameters[0] = {'params': [p for n, p in cur_model.model.named_parameters(
                # ) if not any(nd in n for nd in no_decay + ['word_embeddings'])], 'weight_decay': weight_decay}
                pass

        optimizer_list, scheduler_list = [], []
        optimizer_list.append(
            AdamW(optimizer_grouped_parameters, lr=1e-5, eps=adam_epsilon))
        scheduler_list.append(get_linear_schedule_with_warmup(
            optimizer_list[0], num_warmup_steps=warmup_steps, num_training_steps=t_total))

        if embedding_parameters:
            optimizer_list.append(AdamW(
                embedding_parameters, lr=learning_rate, eps=adam_epsilon))
            scheduler_list.append(get_linear_schedule_with_warmup(
                optimizer_list[0], num_warmup_steps=warmup_steps, num_training_steps=t_total))

        writer = SummaryWriter(log_dir=os.path.join(
            self.config.output_dir, "writer_logs"))

        # TODO
        best_dev_acc, best_dev_f1, best_dev_matt, best_loss = 0.0, 0.0, -1.0, 0.0
        best_global_step, early_stop_epoch, global_step = 0, 0, 0
        prev_loss, tr_loss, logging_loss = 0.0, 0.0, 0.0
        self.model.zero_grad()

        logger.info("dev_data performance before training.")
        dev_scores = self.eval(
            dev_data, eval_config.per_gpu_eval_batch_size, n_gpu, eval_config.metrics)['scores']
        logger.info(dev_scores)

        logger.info("eval_data performance before training.")
        dev_scores = self.eval(
            dev_data, eval_config.per_gpu_eval_batch_size, n_gpu, eval_config.metrics)['scores']
        logger.info(dev_scores)

        train_iterator = trange(int(num_train_epochs), desc="Epoch")
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = {k: t.cuda() for k, t in batch.items()}

                loss = self.task_helper.train_step(
                    batch) if self.task_helper else None
                if loss is None:
                    loss = self.mlm_train_step(batch)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()

                if (step + 1) % gradient_accumulation_steps == 0:
                    # TODO
                    writer.add_scalar(
                        "train_loss", (tr_loss - prev_loss), global_step=global_step)
                    prev_loss = tr_loss

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm)

                    for optimizer, scheduler in zip(optimizer_list, scheduler_list):
                        optimizer.step()
                        scheduler.step()

                    self.model.zero_grad()
                    global_step += 1

                    if logging_steps > 0 and global_step % logging_steps == 0:
                        logs = {}
                        loss_scalar = (tr_loss - logging_loss) / logging_steps
                        learning_rate_scalar = scheduler.get_lr()[0]
                        logs['learning_rate'] = learning_rate_scalar
                        logs['loss'] = loss_scalar
                        logging_loss = tr_loss
                        print(json.dumps({**logs, **{'step': global_step}}))

                    # TODO
                    if global_step % self.config.eval_every_step == 0:
                        dev_scores = self.eval(
                            dev_data, eval_config.per_gpu_eval_batch_size, n_gpu, eval_config.metrics)['scores']
                        is_best = False

                        if self.config.task_name in ["cb", "record", "multirc"]:
                            f1_str = "f1" if self.config.task_name != "cb" else "f1-macro"
                            if dev_scores["acc"] >= best_dev_acc and dev_scores[f1_str] >= best_dev_f1:
                                is_best = True
                                if dev_scores["acc"] > best_dev_acc and dev_scores[f1_str] > best_dev_f1:
                                    early_stop_epoch = 0
                                else:
                                    early_stop_epoch += 1

                                best_dev_acc = dev_scores["acc"]
                                best_dev_f1 = dev_scores[f1_str]
                                best_global_step = global_step
                                best_loss = tr_loss
                                logger.info("best_dev_acc: %.4f | best_dev_f1: %.4f | best_global_step: %d" %
                                            (best_dev_acc, best_dev_f1, best_global_step))

                        elif self.config.task_name in ["rte", "wic", "boolq", "wsc", "copa", "sst-2", "mnli", "sst-5"]:
                            if dev_scores["acc"] >= best_dev_acc:
                                is_best = True
                                if dev_scores["acc"] > best_dev_acc:
                                    early_stop_epoch = 0
                                else:
                                    early_stop_epoch += 1

                                best_dev_acc = dev_scores["acc"]
                                best_global_step = global_step
                                best_loss = tr_loss
                                logger.info("best_dev_acc: %.4f | best_global_step: %d" %
                                            (best_dev_acc, best_global_step))

                        elif self.config.task_name in ["cola"]:
                            if dev_scores["matt"] >= best_dev_matt:
                                is_best = True
                                if dev_scores["matt"] > best_dev_matt:
                                    early_stop_epoch = 0
                                else:
                                    early_stop_epoch += 1

                                best_dev_matt = dev_scores["matt"]
                                best_global_step = global_step
                                best_loss = tr_loss
                                logger.info("best_dev_matt: %.4f | best_global_step: %d" %
                                            (best_dev_matt, best_global_step))

                        # Common operations if achieved best performance on dev set
                        if is_best:
                            self.save(pattern_iter_output_dir)
                            eval_scores = self.eval(
                                dev_data, eval_config.per_gpu_eval_batch_size,
                                n_gpu, eval_config.metrics)['scores']
                            logger.info("Saving trained model at {}...".format(
                                pattern_iter_output_dir))
                            logger.info("eval_data performance:")
                            logger.info(eval_scores)
                        else:
                            early_stop_epoch += 1
                            logger.info(dev_scores)
                            logger.info(early_stop_epoch)

                if 0 < max_steps < global_step or early_stop_epoch >= 10:
                    epoch_iterator.close()
                    break

            if 0 < max_steps < global_step or early_stop_epoch >= 10:
                train_iterator.close()
                break

        # Is this step necessary?
        if self.config.prompt_encoder_type == "inner" and kwargs.get('stage', 0) == 1:
            handle.remove()

        return best_global_step, (best_loss / best_global_step if best_global_step > 0 else -1)

    def eval(self,
             eval_data: List[InputExample],
             per_gpu_eval_batch_size: int = 8,
             n_gpu: int = 1,
             metrics: List[str] = ['acc']) -> Dict:

        eval_dataset = self._generate_dataset(eval_data)
        eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        preds = None
        all_indices, out_label_ids, question_ids = None, None, None
        eval_losses = [0.0]

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = {k: t.cuda() for k, t in batch.items()}
            labels = batch['labels']
            indices = batch['idx']
            with torch.no_grad():

                logits = self.task_helper.eval_step(
                    batch) if self.task_helper else None
                if logits is None:
                    logits = self.mlm_eval_step(batch)

                prediction_scores = logits.float().cuda()
                eval_loss = nn.CrossEntropyLoss()(
                    prediction_scores.view(-1, len(self.config.label_list)), labels.view(-1))
                eval_losses.append(eval_loss.item())

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
                all_indices = indices.detach().cpu().numpy()
                if 'question_idx' in batch:
                    question_ids = batch['question_idx'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, labels.detach().cpu().numpy(), axis=0)
                all_indices = np.append(
                    all_indices, indices.detach().cpu().numpy(), axis=0)
                if 'question_idx' in batch:
                    question_ids = np.append(
                        question_ids, batch['question_idx'].detach().cpu().numpy(), axis=0)

        results = {
            "eval_loss": np.mean(eval_losses),
            'indices': all_indices,
            'logits': preds,
            'labels': out_label_ids,
            'question_ids': question_ids
        }

        return evaluate_results(results, metrics)

    def mlm_train_step(self, labeled_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a MLM training step."""
        inputs = self._generate_default_inputs(labeled_batch)
        mlm_labels, labels = labeled_batch['mlm_labels'], labeled_batch['labels']
        outputs = self.model.model(**inputs)
        if self.config.prompt_encoder_type == "inner":
            prediction_scores = self.encoder.convert_mlm_logits_to_cls_logits(
                mlm_labels, outputs[0])
        else:
            prediction_scores = self.pvp.convert_mlm_logits_to_cls_logits(
                mlm_labels, outputs[0])
        loss = nn.CrossEntropyLoss()(
            prediction_scores.view(-1, len(self.config.label_list)), labels.view(-1))

        return loss

    def mlm_eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a MLM evaluation step."""
        inputs = self._generate_default_inputs(batch)
        outputs = self.model.model(**inputs)

        if self.config.prompt_encoder_type == "inner":
            return self.encoder.convert_mlm_logits_to_cls_logits(batch['mlm_labels'], outputs[0])

        return self.pvp.convert_mlm_logits_to_cls_logits(batch['mlm_labels'], outputs[0])

    def _generate_dataset(self, data: List[InputExample], labelled: bool = True):
        features = self._convert_examples_to_features(data, labelled=labelled)
        # Convert list features to tensors
        feature_dict = {
            'input_ids': torch.tensor([f.input_ids for f in features], dtype=torch.long),
            'attention_mask': torch.tensor([f.attention_mask for f in features], dtype=torch.long),
            'token_type_ids': torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
            'labels': torch.tensor([f.label for f in features], dtype=torch.long),
            'mlm_labels': torch.tensor([f.mlm_labels for f in features], dtype=torch.long),
            'logits': torch.tensor([f.logits for f in features], dtype=torch.float),
            'idx': torch.tensor([f.idx for f in features], dtype=torch.long),
            'block_flag': torch.tensor([f.block_flag for f in features], dtype=torch.long)
        }

        # if self.config.prompt_encoder_type == "inner":
        #     feature_dict['input_ids'] = self.encoder.convert_input_ids(
        #         feature_dict['input_ids'], feature_dict['block_flag'])

        if self.task_helper:
            self.task_helper.add_features_to_dict(features, feature_dict)

        return DictDataset(**feature_dict)

    def _convert_examples_to_features(self, examples: List[InputExample], labelled: bool = True) -> List[InputFeatures]:
        features = []
        for example in examples:
            # Preprocessor for models pretrained using a masked language modeling objective (e.g., BERT).
            input_ids, token_type_ids, block_flag = self.pvp.encode(example)
            attention_mask = [1] * len(input_ids)
            padding_length = self.config.max_seq_length - \
                len(input_ids)

            if padding_length < 0:
                raise ValueError(
                    f"Maximum sequence length is too small, got {len(input_ids)} input ids")

            input_ids = input_ids + \
                ([self.tokenizer.pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
            block_flag = block_flag + ([0] * padding_length)

            assert len(input_ids) == self.config.max_seq_length
            assert len(attention_mask) == self.config.max_seq_length
            assert len(token_type_ids) == self.config.max_seq_length
            assert len(block_flag) == self.config.max_seq_length

            label = self.label_map[example.label] if example.label is not None else -100
            logits = example.logits if example.logits else [-1]

            if labelled:
                mlm_labels = self.pvp.get_mask_positions(input_ids)
            else:
                mlm_labels = [-1] * self.config.max_seq_length

            input_features = InputFeatures(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           token_type_ids=token_type_ids,
                                           label=label,
                                           mlm_labels=mlm_labels,
                                           logits=logits,
                                           idx=example.idx,
                                           block_flag=block_flag)

            # Add meta input features
            if self.task_helper:
                self.task_helper.add_special_input_features(
                    example, input_features)
            features.append(input_features)

        return features

    def _generate_default_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = batch['input_ids']
        bz = batch['input_ids'].shape[0]
        block_flag = batch["block_flag"]
        model = self.model.module if hasattr(
            self.model, 'module') else self.model

        word_embeddings = model.model.get_input_embeddings()
        raw_embeds = word_embeddings(input_ids)

        replace_embeds = model.prompt_embeddings(
            torch.LongTensor(list(range(model.prompt_length))).cuda())
        # [batch_size, prompt_length, embed_size]
        replace_embeds = replace_embeds.unsqueeze(0)

        if self.config.prompt_encoder_type == "lstm":
            # [batch_size, seq_len, 2 * hidden_dim]
            replace_embeds = model.lstm_head(replace_embeds)[0]
            if model.prompt_length == 1:
                replace_embeds = model.mlp_head(replace_embeds)
            else:
                replace_embeds = model.mlp_head(replace_embeds).squeeze()

        elif self.config.prompt_encoder_type == "mlp":
            replace_embeds = model.mlp(replace_embeds)

        elif self.config.prompt_encoder_type == "none":
            replace_embeds = None

        elif self.config.prompt_encoder_type == "inner":
            # assert set(self.encoder.pattern_convert.keys()) == set(input_ids[torch.where(block_flag==1)].tolist())
            replace_embeds = self.encoder.get_replace_embeds(word_embeddings)

        else:
            raise ValueError("unknown prompt_encoder_type.")

        if replace_embeds is not None:  # For normal cases where prompt encoder is not None
            blocked_indices = (block_flag == 1).nonzero().reshape(
                (bz, model.prompt_length, 2))[:, :, 1]

            for bidx in range(bz):
                for i in range(blocked_indices.shape[1]):
                    raw_embeds[bidx, blocked_indices[bidx, i],
                               :] = replace_embeds[i, :]

        inputs = {'inputs_embeds': raw_embeds,
                  'attention_mask': batch['attention_mask']}

        if self.config.model_type in ['bert']:
            inputs['token_type_ids'] = batch['token_type_ids']

        return inputs
