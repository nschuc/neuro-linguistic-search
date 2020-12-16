from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch
import pandas as pd
import numpy as np
import random
import nltk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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
    
def pos_tags(sent):
    return ' '.join([x[1] for x in nltk.pos_tag(nltk.tokenize.word_tokenize(sent))])
    
def create_input(sent) :
    tags = pos_tags(sent)
    return sent + ' POS ' + tags
    
#train data
train_texts, train_labels = [], []

informal_text = open(dir + 'train.informal-formal.informal').readlines()
train_texts.extend([create_input(s) for s  in informal_text])
train_labels.extend([0 for i in range(len(informal_text))])

formal_text = open(dir + 'train.informal-formal.formal').readlines()
train_texts.extend([create_input(s) for s  in formal_text])
train_labels.extend([1 for i in range(len(formal_text))])

data = list(zip(train_texts, train_labels))
random.shuffle(data)
train_texts, train_labels = zip(*data)

#val data
val_texts, val_labels = [], []

informal_text = open(dir + 'valid.informal-formal.informal').readlines()
val_texts.extend([create_input(s) for s  in informal_text])
val_labels.extend([0 for i in range(len(informal_text))])

formal_text = open(dir + 'valid.informal-formal.formal').readlines()
val_texts.extend([create_input(s) for s  in formal_text])
val_labels.extend([1 for i in range(len(formal_text))])


#test data
test_texts, test_labels = [], []

informal_text = open(dir + 'test.informal-formal.informal').readlines()
test_texts.extend([create_input(s) for s  in informal_text])
test_labels.extend([0 for i in range(len(informal_text))])

formal_text = open(dir + 'test.informal-formal.formal').readlines()
test_texts.extend([create_input(s) for s  in formal_text])
test_labels.extend([1 for i in range(len(formal_text))])


print (len(train_texts), len(train_labels), len(val_texts), len(val_labels), len(test_texts), len(test_labels))

#tokenize text data
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

train_dataset = GYAFCDataset(train_encodings, train_labels)
val_dataset = GYAFCDataset(val_encodings, val_labels)
test_dataset = GYAFCDataset(test_encodings, test_labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
}


training_args = TrainingArguments(
    output_dir='roberta-ling-results/',          # output directory
    num_train_epochs=1,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    #warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=25000,
    save_steps=13000,
    fp16=True,
    learning_rate=5e-5
)

model = RobertaForSequenceClassification.from_pretrained('roberta-base', return_dict=True)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,            # evaluation dataset
    compute_metrics=compute_metrics
)

trainer.train()

preds, labels, metrics = trainer.predict(test_dataset)
print('\ntest metrics :', metrics)
preds = ['1' if x[1]>x[0] else '0' for x in preds]
f = open('roberta-ling-results/test_preds.txt', 'w')
f.write('\n'.join(preds))
f.close()
