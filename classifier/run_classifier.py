from transformers import RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

class GYAFCDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings)

texts = open('sentences.txt','r').readlines()
texts = [s.rstrip('\n') for s in texts]

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
encodings = tokenizer(texts, truncation=True, padding=True)
dataset = GYAFCDataset(encodings)

model = RobertaForSequenceClassification.from_pretrained('checkpoint-52000/')
inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)[0].detach().numpy()

preds = ['1' if x[1]>x[0] else '0' for x in outputs]
f = open('preds.txt', 'w')
f.write('\n'.join(preds))
f.close()
