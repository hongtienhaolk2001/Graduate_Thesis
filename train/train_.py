import random

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, AdamW, get_scheduler
from vncorenlp import VnCoreNLP

import loss
from CustomSoftmaxModel import CustomModelSoftmax
from metrics import metric
from preprocessing.NewsPreprocessing import Preprocess
from utils import pred_to_label, update_model
from visualization import Visualization


# Set Seed
seed = 19133022
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# Segmenter input
rdrsegmenter = VnCoreNLP("preprocessing/vncorenlp/VnCoreNLP-1.1.1.jar",
                         annotators="wseg", max_heap_size='-Xmx500m')
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
# Load datasets & Preprocess
preprocess = Preprocess(tokenizer, rdrsegmenter)
tokenized_datasets = preprocess.run(load_dataset('csv', data_files={'train': r"./data/training_data/train_datasets.csv",
                                                                    'test': r"./data/training_data/test_datasets.csv"}))
# Hyper-parameter
num_epochs = 10
learning_rate = 5e-5
batch_size = 32
# Data loader
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=batch_size, collate_fn=data_collator, shuffle=True)
test_dataloader = DataLoader(tokenized_datasets["test"], batch_size=batch_size, collate_fn=data_collator)
# Model
phobert = CustomModelSoftmax("vinai/phobert-base")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
phobert.to(device)
# Optimizer
optimizer = AdamW(phobert.parameters(), lr=learning_rate)
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler('linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)


def evaluation(model, dataloader):
    valid = metric()
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs = {'input_ids': batch['input_ids'].to(device),
                      'attention_mask': batch['attention_mask'].to(device)}
            outputs_classifier, outputs_regressor = model(**inputs)
            classifier_loss = loss.classifier(outputs_classifier, batch['labels_classifier'].to(device).float())
            softmax_loss = loss.softmax(outputs_regressor, batch['labels_regressor'].to(device).float(), device)
            loss = classifier_loss + softmax_loss
            outputs_classifier = outputs_classifier.cpu().numpy()
            outputs_regressor = outputs_regressor.cpu().numpy()
            outputs_regressor = outputs_regressor.argmax(axis=-1) + 1
            outputs = pred_to_label(outputs_classifier, outputs_regressor)

            # update loss
            y_true = batch['labels_regressor'].numpy()
            valid.classifier_loss.update(classifier_loss.item())
            valid.regressor_loss.update(softmax_loss.item())
            valid.loss.update(mix_loss.item())
            valid.acc.update(np.round(outputs), y_true)
            valid.f1_score.update(np.round(outputs), y_true)
            valid.r2_score.update(np.round(outputs), y_true)
    return valid


# save for visualization
train_log = Visualization()
val_log = Visualization()
best_score = -1
for epoch in range(num_epochs):
    train_metrics = metric()
    phobert.train()
    for batch in train_dataloader:
        inputs = {'input_ids': batch['input_ids'].to(device),
                  'attention_mask': batch['attention_mask'].to(device)}
        outputs_classifier, outputs_regressor = phobert(**inputs)
        sigmoid_focal_loss = loss.sigmoid_focal(outputs_classifier, batch['labels_classifier'].to(device).float(),
                                                alpha=-1, gamma=1, reduction='mean')
        softmax_loss = loss.softmax(outputs_regressor, batch['labels_regressor'].to(device).float(), device)
        mix_loss = 10 * sigmoid_focal_loss + softmax_loss
        optimizer.zero_grad()
        mix_loss.backward()
        optimizer.step()
        # with torch.no_grad():
        #     outputs_classifier = outputs_classifier.cpu().numpy()
        #     outputs_regressor = outputs_regressor.cpu().numpy()
        #     outputs_regressor = outputs_regressor.argmax(axis=-1) + 1
        #     outputs = pred_to_label(outputs_classifier, outputs_regressor)
        #
        #     y_true = batch['labels_regressor'].numpy()
        #     train_metrics.sigmoid_focal_loss.update(sigmoid_focal_loss.item())
        #     train_metrics.regressor_loss.update(softmax_loss.item())
        #     train_metrics.loss.update(mix_loss.item())
        #     train_metrics.acc.update(np.round(outputs), y_true)
        #     train_metrics.f1_score.update(np.round(outputs), y_true)
        #     train_metrics.r2_score.update(np.round(outputs), y_true)

    val_metrics = evaluation(phobert, test_dataloader)
    # Save for plot
    train_log.add2log(train_metrics)
    val_log.add2log(val_metrics)
    # Update model
    best_score = update_model(phobert, val_log.log['Score'][-1], best_score)
    # Update learning rate
    lr_scheduler.step()

# print(f'best score: {log.best_score}')
