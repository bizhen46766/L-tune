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

import os
import json
from collections import defaultdict
from typing import List
import numpy as np
import torch
import logging

from model import TransformerModelWrapper
from config import TrainConfig, EvalConfig, WrapperConfig, load_pet_configs
from data_utils import TRAIN_SET, DEV_SET, DEV32_SET, TEST_SET, load_examples, load_metrics
from utils import write_results, save_logits, save_predictions, set_seed, InputExample

logger = logging.getLogger('train')


def train_pet(args):
    # Load configs
    model_config, train_config, eval_config = load_pet_configs(args)

    # Load dataset
    train_data = load_examples(args.task_name, args.data_dir, TRAIN_SET,
                               num_examples=args.train_examples, split_examples_evenly=args.split_examples_evenly)
    eval_data = load_examples(args.task_name, args.data_dir, TEST_SET if args.eval_set == 'test' else DEV_SET,
                              num_examples=args.eval_examples, split_examples_evenly=args.split_examples_evenly)
    dev_data = load_examples(args.task_name, args.data_dir, DEV32_SET,
                             num_examples=args.dev_examples, split_examples_evenly=args.split_examples_evenly)

    results = defaultdict(lambda: defaultdict(list))
    dev_results = defaultdict(lambda: defaultdict(list))
    set_seed(args.seed)

    # Iterates through all patterns and repeat training
    for pattern_id in args.pattern_ids:
        for iteration in range(args.pet_repetitions):

            model_config.pattern_id = pattern_id
            results_dict = {}

            pattern_iter_output_dir = "{}/p{}-i{}".format(
                args.output_dir, pattern_id, iteration)

            if os.path.exists(pattern_iter_output_dir):
                logger.warning(
                    f"Path {pattern_iter_output_dir} already exists, skipping it...")
                continue
            os.makedirs(pattern_iter_output_dir)

            # Init wrapper model
            assert model_config.pattern_id is not None, 'A pattern_id must be set for initializing a new PET model'
            wrapper = TransformerModelWrapper(model_config)

            ##################################################
            # Training
            if args.do_train:

                if not args.two_stage_train:
                    # Single stage training
                    results_dict.update(train_single_model(train_data, eval_data, dev_data, pattern_iter_output_dir,
                                                           wrapper, train_config, eval_config))

                    with open(os.path.join(pattern_iter_output_dir, 'results.txt'), 'w') as fh:
                        fh.write(str(results_dict))

                else:
                    # Two stage training
                    # 1. Only train prompts and label tokens
                    logger.info('Start training stage 1:')
                    results_dict.update(train_single_model(train_data, eval_data, dev_data, pattern_iter_output_dir,
                                                           wrapper, train_config, eval_config, stage=1))
                    with open(os.path.join(pattern_iter_output_dir, 'results_stage1.txt'), 'w') as fh:
                        fh.write(str(results_dict))

                    # 2. Train full model
                    logger.info('Start training stage 2:')
                    results_dict.update(train_single_model(train_data, eval_data, dev_data, pattern_iter_output_dir,
                                                           wrapper, train_config, eval_config, stage=2))

                    with open(os.path.join(pattern_iter_output_dir, 'results_stage2.txt'), 'w') as fh:
                        fh.write(str(results_dict))

                # Save configs
                train_config.save(os.path.join(
                    pattern_iter_output_dir, 'train_config.json'))
                eval_config.save(os.path.join(
                    pattern_iter_output_dir, 'eval_config.json'))
                logger.info("Saving complete")

            ##################################################
            # Evaluation
            if args.do_eval:
                logger.info("Starting evaluation...")

                # if not wrapper:
                wrapper = TransformerModelWrapper.from_pretrained(
                    pattern_iter_output_dir)

                eval_result = wrapper.eval(
                    eval_data, eval_config.per_gpu_eval_batch_size, eval_config.n_gpu, eval_config.metrics)
                dev_result = wrapper.eval(
                    dev_data, eval_config.per_gpu_eval_batch_size, eval_config.n_gpu, eval_config.metrics)

                save_predictions(os.path.join(
                    pattern_iter_output_dir, 'eval_predictions.jsonl'), wrapper, eval_result)
                save_logits(os.path.join(pattern_iter_output_dir,
                            'eval_logits.txt'), eval_result['logits'])

                save_predictions(os.path.join(
                    pattern_iter_output_dir, 'dev_predictions.jsonl'), wrapper, dev_result)
                save_logits(os.path.join(pattern_iter_output_dir,
                            'dev_logits.txt'), dev_result['logits'])

                logger.info(
                    "--- RESULT (pattern_id={}, iteration={}) ---".format(pattern_id, iteration))
                logger.info("eval_results:")
                logger.info(eval_result['scores'])
                logger.info("dev_results:")
                logger.info(dev_result['scores'])

                results_dict['eval_set_after_training'] = eval_result['scores']
                results_dict['dev_set_after_training'] = dev_result['scores']
                with open(os.path.join(pattern_iter_output_dir, 'results.json'), 'w') as fh:
                    json.dump(results_dict, fh)

                for metric, value in eval_result['scores'].items():
                    results[metric][pattern_id].append(value)

                for metric, value in dev_result['scores'].items():
                    dev_results[metric][pattern_id].append(value)

                logger.info("=== OVERALL RESULTS ===")
                write_results(os.path.join(
                    args.output_dir, 'result_test.txt'), results, dev_results)

            # Clear cache
            if not args.do_eval:
                wrapper.model = None
                wrapper = None
                torch.cuda.empty_cache()


def train_single_model(train_data: List[InputExample],
                       eval_data: List[InputExample],
                       dev_data: List[InputExample],
                       pattern_iter_output_dir: str,
                       model: TransformerModelWrapper,
                       config: TrainConfig,
                       eval_config: EvalConfig,
                       **kwargs):
    """
    Train a single model.
    :param model: the model to train
    :param train_data: the training examples to use
    :param config: the training config
    :param eval_config: the evaluation config
    :return: a dictionary containing the global step, average loss and (optionally) results on the train set
    """

    results_dict = {}

    metric_name = load_metrics(model.config.task_name)[0]
    results_dict['train_set_before_training'] = model.eval(
        train_data, eval_config.per_gpu_eval_batch_size,
        eval_config.n_gpu, eval_config.metrics)['scores'][metric_name]

    if not train_data:
        logger.warning('Training method was called without training examples')
    else:
        global_step, tr_loss = model.train(
            pattern_iter_output_dir=pattern_iter_output_dir,
            eval_config=eval_config,
            train_data=train_data,
            dev_data=dev_data,
            eval_data=eval_data,
            per_gpu_train_batch_size=config.per_gpu_train_batch_size,
            n_gpu=config.n_gpu,
            num_train_epochs=config.num_train_epochs,
            max_steps=config.max_steps,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            adam_epsilon=config.adam_epsilon,
            warmup_steps=config.warmup_steps,
            max_grad_norm=config.max_grad_norm,
            alpha=config.alpha,
            **kwargs
        )
        results_dict['global_step'] = global_step
        results_dict['average_loss'] = tr_loss

    model = TransformerModelWrapper.from_pretrained(pattern_iter_output_dir)
    results_dict['train_set_after_training'] = model.eval(
        train_data, eval_config.per_gpu_eval_batch_size,
        eval_config.n_gpu, eval_config.metrics)['scores'][metric_name]

    return results_dict
