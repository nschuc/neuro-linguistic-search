import torch
import numpy as np
import random

ds_path = '/miniscratch/comp_550_project/gyafc/Family_Relationships_comb/'

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
informal_text = open(ds_path + 'train.informal-formal.informal').readlines()
train_texts.extend([s[:-1] for s in informal_text])
train_labels.extend([0 for i in range(len(informal_text))])
formal_text = open(ds_path + 'train.informal-formal.formal').readlines()
train_texts.extend([s[:-1] for s in formal_text])
train_labels.extend([1 for i in range(len(formal_text))])
data = list(zip(train_texts, train_labels))
random.shuffle(data)
train_texts, train_labels = zip(*data)

#val data
val_texts, val_labels = [], []
informal_text = open(ds_path + 'valid.informal-formal.informal').readlines()
val_texts.extend([s[:-1] for s in informal_text])
val_labels.extend([0 for i in range(len(informal_text))])
formal_text = open(ds_path + 'valid.informal-formal.formal').readlines()
val_texts.extend([s[:-1] for s in formal_text])
val_labels.extend([1 for i in range(len(formal_text))])
data = list(zip(val_texts, val_labels))
random.shuffle(data)
val_texts, val_labels = zip(*data)

print (len(train_texts), len(train_labels), len(val_texts), len(val_labels))

ds_path = '/miniscratch/comp_550_project/gyafc/Family_Relationships_comb/'

from ling.ling_feature import LingFeatureMaker, make_model_inp

formal_train_path, informal_train_path = ds_path + 'train.informal-formal.formal', ds_path + 'train.informal-formal.informal'
formal_val_path, informal_val_path = ds_path + 'valid.informal-formal.formal', ds_path + 'valid.informal-formal.informal'
formal_test_path, informal_test_path = ds_path + 'test.informal-formal.formal', ds_path + 'test.informal-formal.informal'

import tqdm
import os

def build_linguistic_features(lfm, texts):
    feats = []
    for line in tqdm.tqdm(texts):
        x1 = np.mean(lfm.word_counts(line))
        x2 = np.mean(lfm.word_length(line))
        x3 = np.mean(lfm.latinate(line))
        x4 = lfm.has_emoticon(line)
        x5 = lfm.embed_pos(line)
        feats.append(np.array([x1, x2, x3, x4, x5]))
    return feats

if os.path.exists('ling-feats/train-feats.npy'):
    with open('ling-feats/train-feats.npy', 'rb') as f:
        train_ling_feats = np.load(f)
else:
    train_ling_feats = build_linguistic_features(lfm, train_texts)
    
if os.path.exists('ling-feats/val-feats.npy'):
    with open('ling-feats/val-feats.npy', 'rb') as f:
        val_ling_feats = np.load(f)
else:
    val_ling_feats = build_linguistic_features(lfm, val_texts)

class LinguisticFeaturesDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, linguistic_feats, labels, ):
        self.encodings = encodings
        self.labels = labels
        self.linguistic_feats = linguistic_feats

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['ling_feats'] = torch.tensor(self.linguistic_feats[idx]).half()

        return item

    def __len__(self):
        return len(self.labels)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import RobertaTokenizer, RobertaModel, BertPreTrainedModel, RobertaForSequenceClassification, Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

LING_FEAT_DIM = 5

class LingBertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(LingBertaClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size + LING_FEAT_DIM, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, ling_feats, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = torch.cat([x, ling_feats], dim=1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class LingBertaFormalityClassifier(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = LingBertaClassificationHead(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        ling_feats=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output, ling_feats)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

#tokenize import nltktext data
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

train_dataset = LinguisticFeaturesDataset(train_encodings, train_ling_feats, train_labels)
val_dataset = LinguisticFeaturesDataset(val_encodings, val_ling_feats, val_labels)

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

model = LingBertaFormalityClassifier.from_pretrained('roberta-base', return_dict=True)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()
