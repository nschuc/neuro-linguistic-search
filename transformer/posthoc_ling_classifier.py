import torch
import numpy as np
import random
import typer
from ling.ling_feature import LingFeatureMaker, make_model_inp
import tqdm
import os

ds_path = '/home/toolkit/gyafc/'

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

def load_data(split: str, shuffle: bool = False):
    #train data
    texts, labels = [], []
    informal_text = open(ds_path + f'{split}.informal-formal.informal').readlines()
    texts.extend([s[:-1] for s in informal_text])
    labels.extend([0 for i in range(len(informal_text))])
    formal_text = open(ds_path + f'{split}.informal-formal.formal').readlines()
    texts.extend([s[:-1] for s in formal_text])
    labels.extend([1 for i in range(len(formal_text))])

    if shuffle:
        data = list(zip(texts, labels))
        random.shuffle(data)
        texts, labels = zip(*data)

    return texts, labels



formal_train_path, informal_train_path = ds_path + 'train.informal-formal.formal', ds_path + 'train.informal-formal.informal'
formal_val_path, informal_val_path = ds_path + 'valid.informal-formal.formal', ds_path + 'valid.informal-formal.informal'
formal_test_path, informal_test_path = ds_path + 'test.informal-formal.formal', ds_path + 'test.informal-formal.informal'


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

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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


def main(name: str = "lingberta", train: bool = False, evaluate: bool = False, test: bool = False, ckpt: str = 'checkpoint-52000', pred_path: str = "preds.txt"):

    if train or evaluate:
        train_texts, train_labels = load_data('train', shuffle=True)
        val_texts, val_labels = load_data('valid')

        if os.path.exists(f'ling-feats/train-feats.npy'):
            with open('ling-feats/train-feats.npy', 'rb') as f:
                train_ling_feats = np.load(f)
        else:
            lfm = LingFeatureMaker(formal_train_path, informal_train_path, use_pos_lm=True)
            train_ling_feats = build_linguistic_features(lfm, train_texts)
            
        if os.path.exists('ling-feats/val-feats.npy'):
            with open('ling-feats/val-feats.npy', 'rb') as f:
                val_ling_feats = np.load(f)
        else:
            lfm = LingFeatureMaker(formal_val_path, informal_val_path, use_pos_lm=True)
            val_ling_feats = build_linguistic_features(lfm, val_texts)


        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        train_encodings = tokenizer(train_texts, truncation=True, padding=True)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True)

        train_dataset = LinguisticFeaturesDataset(train_encodings, train_ling_feats, train_labels)
        val_dataset = LinguisticFeaturesDataset(val_encodings, val_ling_feats, val_labels)


    if train:
        model = LingBertaFormalityClassifier.from_pretrained('roberta-base', return_dict=True)

    if evaluate or test:
        model = LingBertaFormalityClassifier.from_pretrained(f'roberta-results/{ckpt}', return_dict=True)


    training_args = TrainingArguments(
        output_dir=name,          # output directory
        num_train_epochs=1,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,   # batch size for evaluation
        # warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=3000,
        evaluation_strategy="steps",
        fp16=True,
        learning_rate=5e-5
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if train or evaluate else None,
        eval_dataset=val_dataset if train or evaluate  else None,
        compute_metrics=compute_metrics
    )

    if train:
        trainer.train()

    if evaluate:
        trainer.evaluate()


    if test:
        test_texts, test_labels = load_data('test')

        if os.path.exists(f'ling-feats/test-feats.npy'):
            with open('ling-feats/test-feats.npy', 'rb') as f:
                test_ling_feats = np.load(f)
        else:
            lfm = LingFeatureMaker(formal_test_path, informal_test_path, use_pos_lm=True)
            test_ling_feats = build_linguistic_features(lfm, test_texts)
            
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        test_encodings = tokenizer(test_texts, truncation=True, padding=True)
        test_dataset = LinguisticFeaturesDataset(test_encodings, test_ling_feats, test_labels)

        preds, labels, metrics = trainer.predict(test_dataset)
        print(metrics)
        preds = ['1' if x[1]>x[0] else '0' for x in preds]
        f = open('preds.txt', 'w')
        f.write('\n'.join(preds))
        f.close()


if __name__ == "__main__":
    typer.run(main)

