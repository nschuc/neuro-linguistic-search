#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch
import pandas as pd
import numpy as np
import random


# ## prepare data for transformers

# In[2]:


dir = '/miniscratch/comp_550_project/gyafc/Family_Relationships_comb/'

class GYAFCDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

#train data
train_texts, train_labels = [], []
informal_text = open(dir + 'train.informal-formal.informal').readlines()
train_texts.extend([s[:-1] for s in informal_text])
train_labels.extend([0 for i in range(len(informal_text))])
formal_text = open(dir + 'train.informal-formal.formal').readlines()
train_texts.extend([s[:-1] for s in formal_text])
train_labels.extend([1 for i in range(len(formal_text))])
data = list(zip(train_texts, train_labels))
random.shuffle(data)
train_texts, train_labels = zip(*data)

#val data
val_texts, val_labels = [], []
informal_text = open(dir + 'valid.informal-formal.informal').readlines()
val_texts.extend([s[:-1] for s in informal_text])
val_labels.extend([0 for i in range(len(informal_text))])
formal_text = open(dir + 'valid.informal-formal.formal').readlines()
val_texts.extend([s[:-1] for s in formal_text])
val_labels.extend([1 for i in range(len(formal_text))])
data = list(zip(val_texts, val_labels))
random.shuffle(data)
val_texts, val_labels = zip(*data)

print (len(train_texts), len(train_labels), len(val_texts), len(val_labels))

#tokenize text data
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

train_dataset = GYAFCDataset(train_encodings, train_labels)
val_dataset = GYAFCDataset(val_encodings, val_labels)

# In[3]:


training_args = TrainingArguments(
    output_dir='roberta-results/',          # output directory
    num_train_epochs=2,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=8,   # batch size for evaluation
    #warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=25000,
    save_steps=13000,
    fp16=True,
    learning_rate=1e-6
)

model = RobertaForSequenceClassification.from_pretrained('roberta-base', return_dict=True)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()
