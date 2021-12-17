
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from torch import nn
from torch.optim import Adam
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback


#load pickle of labels
LABELS_CSV_PATH = "/data/join_result.csv"
topics = pd.read_csv(LABELS_CSV_PATH)
topics = topics.drop(topics[topics['label'] == 0].index)


# Division by topic:
# 1: External Relations
# 2: Freedom and Democracy
# 3: Political System
# 4: Economy
# 5: Welfare and quality of life
# 6: Fabric of Society
# 7: Social Groups

for i in range(1, 8):
    topics.loc[ topics['label']//100 == i , 'label'] = i - 1

def clean_txt(txt):
    txt = re.sub("'","",txt)
    txt = re.sub("(\\W)+"," ",txt)
    return txt

topics['sentence'] = topics['sentence'].apply(clean_txt)

# Read data
data = topics

# Define pretrained tokenizer and model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=7)

# ----- 1. Preprocess data -----#
# Preprocess data
X = list(data["sentence"])
y = list(data["label"])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)

# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

train_dataset = Dataset(X_train_tokenized, y_train)
val_dataset = Dataset(X_val_tokenized, y_val)

# ----- 2. Fine-tune pretrained model -----#
# Define Trainer parameters
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy}

# Define Trainer
args = TrainingArguments(
    output_dir="output",
    evaluation_strategy="steps",
    eval_steps=500,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    seed=0,
    load_best_model_at_end=True,
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=6)],
)

# Train pre-trained model
trainer.train()

