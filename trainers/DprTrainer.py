
import os
import logging
import time
import torch
import numpy as np

import random
from data.datastructures.hyperparameters.dpr import DenseHyperParams
from data.loaders.DprDataLoader import DprDataLoader
from transformers import (DPRContextEncoderTokenizerFast,
                          DPRQuestionEncoderTokenizerFast)
from methods.ir.dense.dpr.models.bi_encoder import BiEncoder
from trainers.utils import get_optimizer

logger = logging.getLogger(__name__)


class BiEncoderTrainer():
    def __init__(self, config: DenseHyperParams):
        self.args = config

        self.query_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(
            self.args.query_encoder_path)
        self.document_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(
            self.args.document_encoder_path)
        self.model = BiEncoder(self.args)
        self.model.to(self.args.device)
        self.model.train()
        self.optimizer = get_optimizer(
            self.args, self.args.epochs * self.args.num_instances, self.model)
        self.loader = DprDataLoader(self.args, self.query_tokenizer, self.document_tokenizer,
                                      self.args.train_dir)

        self.last_save_time = time.time()

        self.first_batch_num = 0
        self.args.set_seed()


    def save_state(self, save_to_path, batch_num):
        logger.info(f'saving current state to {save_to_path}')

        state_dict = {}

        state_dict['epoch'] = self.loader.on_epoch
        state_dict['batch'] = batch_num
        state_dict['qry_model_state_dict'] = self.model.query_model.state_dict()
        state_dict['ctx_model_state_dict'] = self.model.document_model.state_dict()
        state_dict['optimizer_state_dict'] = self.optimizer.optimizer.state_dict()
        state_dict['scheduler_state_dict'] = self.optimizer.scheduler.state_dict()

        torch.save(state_dict, save_to_path)
        logger.info(f'saved current state to {save_to_path}')

    def load_checkpoint(self, load_from_path):
        logger.info(f'loading checkpoint from {load_from_path}')

        checkpoint = torch.load(load_from_path, map_location='cpu')

        # checkpoint based on the last batch in the epoch
        if checkpoint['batch'] + 1 >= self.batches.num_batches:
            checkpoint['batch'] = -1
            checkpoint['epoch'] += 1

        while self.loader.on_epoch < checkpoint['epoch']:
            self.batches = self.loader.get_dataloader()

        self.loader.on_epoch = checkpoint['epoch']
        self.first_batch_num = checkpoint['batch'] + 1

        self.model.qry_model.load_state_dict(
            checkpoint['qry_model_state_dict'])  # , strict=False)
        self.model.ctx_model.load_state_dict(
            checkpoint['ctx_model_state_dict'])  # , strict=False)

        self.optimizer.optimizer.load_state_dict(
            checkpoint['optimizer_state_dict'])
        self.optimizer.scheduler.load_state_dict(
            checkpoint['scheduler_state_dict'])

        torch.set_rng_state(checkpoint['torch_rng_state'].to(
            torch.get_rng_state().device))
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.to(torch.cuda.get_rng_state_all()[
                                         pos].device) for pos, state in enumerate(checkpoint['torch_cuda_rng_states'])])
        np.random.set_state(checkpoint['np_rng_state'])
        random.setstate(checkpoint['python_rng_state'])

        logger.info(f'loaded checkpoint from {load_from_path}')

    def save_tokenizers(self):
        self.ctx_tokenizer.save_pretrained(
            os.path.join(self.args.output_dir, 'ctx_encoder'))
        self.qry_tokenizer.save_pretrained(
            os.path.join(self.args.output_dir, 'qry_encoder'))

    def train(self):
        self.save_tokenizers()
        while True:
            self.batches = self.loader.get_dataloader()
            if not self.optimizer.should_continue() or self.batches is None:
                if not self.optimizer.should_continue():
                    logger.info(
                        f'Breaking, self.optimizer.should_continue() is False')
                if self.batches is None:
                    logger.info(f'Breaking, self.batches is None')
                break
            logger.info(f'len(self.batches) {len(self.batches)}')

            if self.args.resume_from_checkpoint != '':
                if self.args.world_size != 1:
                    raise NotImplementedError(
                        f'Resuming training from a checkpoint is not supported (yet) for world_size != 1.')

                self.load_checkpoint(self.args.resume_from_checkpoint)
                self.args.resume_from_checkpoint = ''

            for batch_num in list(range(self.first_batch_num, self.batches.num_batches)):
                batch = self.batches[batch_num]
                loss, accuracy = self.optimizer.model(
                    **self.loader.batch_dict(batch))
                if self.args.log_all_losses:
                    logger.info(f'batch_num: {batch_num}, {loss}, {accuracy}')
                self.optimizer.step_loss(loss, accuracy=accuracy)
                if not self.optimizer.should_continue():
                    break

                if self.args.log_every_num_batches > 0 and (batch_num % self.args.log_every_num_batches) == 0:
                    logger.info(f'batch_num: {batch_num}')
                    self.optimizer.optimizer_report()

                if time.time() - self.last_save_time > 60 * 60 or \
                        (self.args.save_every_num_batches > 0 and batch_num > 0 and (batch_num % self.args.save_every_num_batches) == 0):
                    # save once an hour or after each "save_every_num_batches" (whichever is more frequent)
                    self.save_checkpoint(os.path.join(
                        self.args.output_dir, "latest_checkpoint"), batch_num)
                    model_to_save = (self.optimizer.model.module if hasattr(
                        self.optimizer.model, "module") else self.optimizer.model)
                    logger.info(f'saving to {self.args.output_dir}')
                    model_to_save.save(self.args.output_dir)
                    self.last_save_time = time.time()

            # save after each epoch
            self.save_checkpoint(os.path.join(
                self.args.output_dir, "latest_checkpoint"), batch_num)
            model_to_save = (self.optimizer.model.module if hasattr(
                self.optimizer.model, "module") else self.optimizer.model)
            logger.info(f'saving to {self.args.output_dir}')
            model_to_save.save(self.args.output_dir)
            self.last_save_time = time.time()
            self.first_batch_num = 0

        # save after running out of files or target num_instances
        logger.info(f'All done')
        self.optimizer.reporting.display()
        model_to_save = (self.optimizer.model.module if hasattr(
            self.optimizer.model, "module") else self.optimizer.model)
        logger.info(f'saving to {self.args.output_dir}')
        model_to_save.save(self.args.output_dir)
        logger.info(f'Took {self.optimizer.reporting.elapsed_time_str()}')


def main():
    trainer = BiEncoderTrainer()
    trainer.train()


if __name__ == "__main__":
    main()
