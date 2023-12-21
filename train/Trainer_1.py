import random
import time

import numpy as np
import torch
from transformers import AdamW, get_scheduler

import loss
from metrics import ScalarMetric, F1_score, R2_score
from utils import get_y

# Set Seed
seed = 19133022
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class Trainer:
    def __init__(self, model, train_dataloader, valid_dataloader):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.model.to(self.device)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

    def update_model(self, best_score, best_score_eval):
        if best_score < best_score_eval:
            best_score = best_score_eval
            torch.save(self.model.state_dict(), "weights/model.pt")
            print(f"update model with score {best_score}")
        return best_score

    def train_epoch(self, optimizer, criterion):
        epoch_f1 = F1_score()
        epoch_r2 = R2_score()
        epoch_loss = ScalarMetric()
        self.model.train()
        for batch in self.train_dataloader:
            print(f"batch['input_ids'] {batch['input_ids']}")
            print(f"batch['attention_mask'] {batch['attention_mask']}")
            inputs = {'input_ids': batch['input_ids'].to(self.device),
                      'attention_mask': batch['attention_mask'].to(self.device)}
            outputs_classifier, outputs_regressor = self.model(**inputs)
            batch_loss = criterion(batch, outputs_classifier, outputs_regressor, self.device)
            optimizer.zero_grad()
            # Backward pass and optimization
            batch_loss.backward()
            optimizer.step()
            with torch.no_grad():
                epoch_loss.update(batch_loss)
                y_pred, y_true = get_y(batch, outputs_classifier, outputs_regressor)
                # score
                epoch_f1.update(y_pred, y_true)
                epoch_r2.update(y_pred, y_true)
        score = (epoch_f1.compute() * epoch_r2.compute()).sum() * 1 / 6
        return score, epoch_loss.compute()

    def evaluate_epoch(self, criterion):
        epoch_loss = ScalarMetric()
        epoch_f1 = F1_score()
        epoch_r2 = R2_score()
        self.model.eval()
        with torch.no_grad():
            for batch in self.valid_dataloader:
                inputs = {'input_ids': batch['input_ids'].to(self.device),
                          'attention_mask': batch['attention_mask'].to(self.device)}
                outputs_classifier, outputs_regressor = self.model(**inputs)
                # loss
                epoch_loss.update(criterion(batch, outputs_classifier, outputs_regressor, self.device))
                y_pred, y_true = get_y(batch, outputs_classifier, outputs_regressor)
                # score
                epoch_f1.update(y_pred, y_true)
                epoch_r2.update(y_pred, y_true)
        score = (epoch_f1.compute() * epoch_r2.compute()).sum() * 1 / 6
        return score, epoch_loss.compute()

    def training(self, num_epochs=15, learning_rate=5e-5):
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        lr_scheduler = get_scheduler('linear', optimizer=optimizer,
                                     num_warmup_steps=0,
                                     num_training_steps=num_epochs * len(self.train_dataloader))
        train_f1_hist, eval_f1_hist, train_loss_hist, eval_loss_hist = [], [], [], []
        times = []
        best_score = -1
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            train_f1, train_loss = self.train_epoch(optimizer, criterion=loss.custom_loss_1)
            train_f1_hist.append(train_f1)
            train_loss_hist.append(train_loss)
            # Evaluation
            eval_f1, eval_loss = self.evaluate_epoch(criterion=loss.custom_loss_2)
            eval_f1_hist.append(eval_f1)
            eval_loss_hist.append(eval_loss)
            # Save best model
            best_score = self.update_model(best_score, eval_f1)
            times.append(time.time() - epoch_start_time)
            print("-" * 59)
            print(
                "| End of epoch {:3d} | Time: {:5.2f}s | Train F1 {:8.3f} | Train Loss {:8.3f} "
                "| Valid F1 {:8.3f} | Valid Loss {:8.3f} ".format(
                    epoch, time.time() - epoch_start_time, train_f1, train_loss, eval_f1, eval_loss
                )
            )
            print("-" * 59)
            lr_scheduler.step()
        return train_f1_hist, eval_f1_hist, train_loss_hist, eval_loss_hist


# if __name__ == '__main__':
#     from vncorenlp import VnCoreNLP
#     from transformers import AutoTokenizer, DataCollatorWithPadding
#     from preprocessing.NewsPreprocessing import Preprocess
#     from datasets import load_dataset
#     from torch.utils.data import DataLoader
#     from model import Model_1
#     rdrsegmenter = VnCoreNLP("preprocessing/vncorenlp/VnCoreNLP-1.1.1.jar",
#                              annotators="wseg", max_heap_size='-Xmx500m')
#     tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
#     preprocess = Preprocess(tokenizer, rdrsegmenter)
#     tokenized_datasets = preprocess.run(
#         load_dataset('csv', data_files={'train': r"./data/training_data/train_datasets.csv",
#                                         'test': r"./data/training_data/test_datasets.csv"}))
#     data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
#     train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=32, collate_fn=data_collator, shuffle=True)
#     valid_dataloader = DataLoader(tokenized_datasets["test"], batch_size=32, collate_fn=data_collator)
#     trainer = Trainer(model=Model_1("vinai/phobert-base"),
#                       train_dataloader=train_dataloader,
#                       valid_dataloader=valid_dataloader, )
#     train_f1_viz, eval_f1_viz, train_loss_viz, eval_loss_viz = trainer.training()
