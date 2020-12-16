
## Running Models

### Transformer RoBERTa classifier
#### With Raw Text as Input
```bash
cd transformer
python roberta.py
```
It acheives around 91.6% accuracy on test set.

#### With Raw Text + Linguistic Features as Input

The input is created by appending POS tag sequence to original sentence. E.g. ```COMP550 is a very informative course <POS> CD VBZ DT RB JJ NN```

```bash
cd transformer
python ling-roberta.py
```
It acheives around 92.3% accuracy on test set, an improvement over previous model.

### RNN LSTM classifier
#### With Raw Text as Input
```bash
cd lstm_cls
python main.py --data <data_dir> --glove_path <path to glove embeddings> --epochs 3 --cuda --save_dir outputs
```
It acheives around 91.92% accuracy on test set.

#### With Raw Text + Linguistic Features as Input

The input is created by appending POS tag sequence to original sentence. E.g. ```COMP550 is a very informative course <POS> CD VBZ DT RB JJ NN```

```bash
cd lstm_cls
python main.py --data <data_dir> --glove_path <path to glove embeddings> --epochs 3 --cuda --save_dir outputs --use_ling_features True
```
It acheives around 91.95% accuracy on test set, a marginal improvement over previous model.
