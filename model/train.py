import random
from transformers import AutoTokenizer
from vncorenlp import VnCoreNLP
from CustomSoftmaxModel import CustomModelSoftmax
from loss import *
from metrics import *
from preprocessing.NewsPreprocessing import Preprocess
from utils import *
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import DataCollatorWithPadding, AdamW, get_scheduler


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

# Load datasets
dataset = load_dataset('csv',
                       data_files={'train': r"./data/training_data/train_datasets.csv",
                                   'test': r"./data/training_data/test_datasets.csv"})

# Preprocess
preprocess = Preprocess(tokenizer, rdrsegmenter)
tokenized_datasets = preprocess.run(dataset)
# Data loader
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = DataLoader(tokenized_datasets["train"],
                              batch_size=32,
                              collate_fn=data_collator,
                              shuffle=True)

test_dataloader = DataLoader(tokenized_datasets["test"],
                             batch_size=32,
                             collate_fn=data_collator)

# Model
model = CustomModelSoftmax("vinai/phobert-base")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 10
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler('linear',
                             optimizer=optimizer,
                             num_warmup_steps=0,
                             num_training_steps=num_training_steps)

# Training
pb_train = tqdm(range(num_training_steps))
pb_test = tqdm(range(num_epochs * len(test_dataloader)))
best_score = -1

for epoch in range(num_epochs):
    train_loss = 0
    val_loss = 0

    # Train
    model.train()
    for batch in train_dataloader:
        inputs = {'input_ids': batch['input_ids'].to(device),
                  'attention_mask': batch['attention_mask'].to(device)}
        outputs_classifier, outputs_regressor = model(**inputs)
        loss1 = sigmoid_focal_loss(outputs_classifier, batch['labels_classifier'].to(device).float(),
                                   alpha=-1, gamma=1, reduction='mean')
        loss2 = loss_softmax(outputs_regressor, batch['labels_regressor'].to(device).float(), device)
        loss = 10 * loss1 + loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        pb_train.update(1)
        pb_train.set_postfix(loss_classifier=loss1.item(), loss_regressor=loss2.item(), loss=loss.item())
        train_loss += loss.item() / len(train_dataloader)
    print("Train Loss:", train_loss)

    val_loss = ScalarMetric()
    val_loss_classifier = ScalarMetric()
    val_loss_regressor = ScalarMetric()
    val_acc = AccuracyMetric()
    val_f1_score = F1_score()
    val_r2_score = R2_score()
    num = 0
    correct = 0
    result = None

    model.eval()
    for batch in test_dataloader:
        inputs = {'input_ids': batch['input_ids'].to(device),
                  'attention_mask': batch['attention_mask'].to(device)}
        with torch.no_grad():
            outputs_classifier, outputs_regressor = model(**inputs)
            loss1 = loss_classifier(outputs_classifier, batch['labels_classifier'].to(device).float())
            loss2 = loss_softmax(outputs_regressor, batch['labels_regressor'].to(device).float(), device)
            loss = loss1 + loss2
            outputs_classifier = outputs_classifier.cpu().numpy()
            outputs_regressor = outputs_regressor.cpu().numpy()
            outputs_regressor = outputs_regressor.argmax(axis=-1) + 1
            y_true = batch['labels_regressor'].numpy()
            outputs = pred_to_label(outputs_classifier, outputs_regressor)
            result = np.concatenate([result, np.round(outputs)], axis=0) if result is not None else np.round(outputs)

            # update loss
            val_loss_classifier.update(loss1.item())
            val_loss_regressor.update(loss2.item())
            val_loss.update(loss.item())
            val_acc.update(np.round(outputs), y_true)
            val_f1_score.update(np.round(outputs), y_true)
            val_r2_score.update(np.round(outputs), y_true)
            pb_test.update(1)

    f1_score = val_f1_score.compute()
    r2_score = val_r2_score.compute()
    final_score = (f1_score * r2_score).sum() * 1 / 6

    if final_score > best_score:
        best_score = final_score
        torch.save(model.state_dict(), "weights/model.pt")

    print(f"""
        Test Loss: {val_loss.compute()}
        Loss Classifier: {val_loss_classifier.compute()}
        Loss Regressor: {val_loss_regressor.compute()}
        Acc: {val_acc.compute()}
        F1 score: {f1_score}
        R2 score: {r2_score}
        Final_score: {final_score}
        Best score: {best_score}""")

print('Train complete')

# class train:
#     def __init__(self):
#         # Segmenter input
#         self.rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
#         # Tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
#         # Load datasets
#         self.data_files = {'train': "../data/training_data/train_datasets.csv",
#                            'test': "../data/training_data/test_datasets.csv"}
#
#         self.dataset = load_dataset('csv', data_files=data_files)
#
#         # Preprocess
#         self.tokenized_datasets = Preprocess(tokenizer, rdrsegmenter).run(dataset)
#
#         # Data loader
#         self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
#
#         self.train_dataloader = DataLoader(tokenized_datasets["train"],
#                                            batch_size=32,
#                                            collate_fn=data_collator,
#                                            shuffle=True)
#
#         self.test_dataloader = DataLoader(tokenized_datasets["test"],
#                                           batch_size=32,
#                                           collate_fn=data_collator)
#
#         # Model
#         self.model = CustomModelSoftmax("vinai/phobert-base")
#         self.model.to('cuda' if torch.cuda.is_available() else 'cpu')  # device
#
#
#
#     def train(self, epochs=10):
#         num_training_steps = epochs * len(train_dataloader)
#         # Optimizer
#         optimizer = AdamW(model.parameters(), lr=5e-5)
#         self.lr_scheduler = get_scheduler('linear',
#                                           optimizer=optimizer,
#                                           num_warmup_steps=0,
#                                           num_training_steps=num_training_steps)
#
#     def evaluation(self):
#         pass
#
#     def __str__(self):
#         return f"""
#         Test Loss: {val_loss.compute()}
#         Loss Classifier: {val_loss_classifier.compute()}
#         Loss Regressor: {val_loss_regressor.compute()}
#         Acc: {val_acc.compute()}
#         F1 score: {f1_score}
#         R2 score: {r2_score}
#         Final_score: {final_score}
#         Best score: {best_score}"""
#
#
# train = train()

