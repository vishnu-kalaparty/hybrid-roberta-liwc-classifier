# Emotion-Aware Academic Writing Assistant

A hybrid model combining RoBERTa embeddings with LIWC-like lexicon features for:
- **Emotion Detection** (6 classes: Positive, Negative, Neutral, Confident, Frustrated, Anxious)
- **Formality Classification** (Formal vs Informal)

## Project Structure

The codebase is organized into modular components:

```
hybrid-roberta-liwc-classifier/
├── main.py                 # Main entry point
├── config.py               # Configuration and hyperparameters
├── .gitignore             # Git ignore rules
├── models/                 # Model definitions
│   ├── __init__.py
│   └── hybrid_model.py     # HybridRoBERTaLIWCModel
├── data/                   # Data loading and preprocessing
│   ├── __init__.py
│   ├── dataset.py          # TextDataset class
│   └── data_loader.py      # Data loading functions
├── features/               # Feature extraction
│   ├── __init__.py
│   └── liwc_extractor.py   # LIWCFeatureExtractor
└── training/               # Training and evaluation
    ├── __init__.py
    └── trainer.py          # Training functions
```

## Setup

### 1. Install Dependencies

```bash
cd hybrid-roberta-liwc-classifier

python -m venv .venv
source .venv/bin/activate  # macOS / Linux

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers pandas numpy scikit-learn tqdm
```

### 2. Data Structure

The code expects data files in the parent directory (or set `DATA_DIR` environment variable):

```text
parent_directory/
├── hybrid-roberta-liwc-classifier/  # This repo
│   ├── main.py
│   ├── config.py
│   └── ...
├── FormalityDataset.csv
└── GoEmotions/
    ├── goemotions_1.csv
    ├── goemotions_2.csv
    └── goemotions_3.csv
```

**Note:** By default, `DATA_DIR` points to the parent directory. To use a different location:

```bash
export DATA_DIR="/absolute/path/to/your/data"
```

## Usage

```bash
cd hybrid-roberta-liwc-classifier
source .venv/bin/activate  # if using the venv
python main.py
```

You'll see:
- Training + validation logs for emotion and formality
- Final test metrics + confusion matrices
- A small demo with predictions on a few example sentences

Note: on CPU it can be slow; GPU or M1/M2 (MPS) is much faster.

---

## Code Organization

### Configuration (`config.py`)
- All hyperparameters and settings
- Device configuration
- Label mappings

### Models (`models/hybrid_model.py`)
- `HybridRoBERTaLIWCModel`: Combines RoBERTa embeddings with LIWC features

### Data (`data/`)
- `dataset.py`: PyTorch Dataset class for text classification
- `data_loader.py`: Functions to load and preprocess GoEmotions and Formality datasets

### Features (`features/liwc_extractor.py`)
- `LIWCFeatureExtractor`: Extracts lexicon-based linguistic features

### Training (`training/trainer.py`)
- `train_epoch`: Training for one epoch
- `evaluate`: Model evaluation
- `train_model`: Full training loop with early stopping

### Main (`main.py`)
- Orchestrates training and evaluation for both tasks
- Handles model initialization, data splitting, and demo predictions
