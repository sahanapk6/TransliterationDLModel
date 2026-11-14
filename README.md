# Indic-to-English Transliteration

Neural machine transliteration between Indic scripts and the Roman alphabet, built for CS 685 (Advanced NLP) at UMass Amherst, Spring 2023. Two sequence-to-sequence models are trained and their predictions combined with an ensemble.

The full write-up is in [cs685_s23_final_report.pdf](cs685_s23_final_report.pdf).

## Task and data

Transliteration maps a word from one script to another while preserving pronunciation. This project uses the [Aksharantar](https://huggingface.co/datasets/ai4bharat/Aksharantar) dataset and focuses on two South Indian languages with long, phonetically rich words:

| Language | Train | Validation | Test |
| --- | --- | --- | --- |
| Kannada | 2,906,728 | 7,025 | 11,380 |
| Telugu | 2,429,562 | 7,681 | 10,260 |

Preprocessing: character-level tokenisation, vocabulary creation, special tokens, padding, batching and masking.

## Models

- **Transformer Seq2Seq:** encoder-decoder with self-attention, which handles long-range dependencies within a word.
- **Convolutional Seq2Seq:** convolutional encoder-decoder with attention, which captures local character patterns.
- **Ensemble:** a distance-based combination of the two models' predictions (`ensemble.py`).

Hyperparameters: embedding size 100, 4 attention heads, 6 encoder and 6 decoder layers, hidden size 200, feed-forward size 2048, learning rate 1e-3, dropout 0.1.

See the report for accuracy figures and error analysis.

## Code

| File | Purpose |
| --- | --- |
| `main.py` | Entry point: builds the model, loads data, trains and evaluates |
| `dataloader.py` | Loading, tokenisation, batching |
| `transformer.py` | Model definitions: embeddings, Transformer and convolutional encoder/decoder layers |
| `decoder.py` | Decoding of output sequences |
| `trainer.py` | Training loop, optimiser, learning-rate schedule, early stopping |
| `ensemble.py` | Combines the two models' outputs |
| `testing.py` | Evaluation script |
| `environment.yml` | Conda environment |

## Run

```sh
conda env create -f environment.yml

python main.py --train [train_file] --test [test_file] --dev [dev_file] --arch [architecture]
```
