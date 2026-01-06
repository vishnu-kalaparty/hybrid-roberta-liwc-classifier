"""
Configuration file for Hybrid Model training.
Contains all hyperparameters and settings.
"""

import os
import torch

# Device configuration
device = torch.device('mps' if torch.backends.mps.is_available() else 
                      'cuda' if torch.cuda.is_available() else 'cpu')

# Data configuration
# Defaults to parent directory (NLP_Proj) where data files are located
# Override with environment variable: export DATA_DIR="/path/to/data"
DATA_DIR = os.environ.get('DATA_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Training hyperparameters
BATCH_SIZE = 96
GRADIENT_ACCUMULATION_STEPS = 1
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
MAX_LENGTH = 96
PATIENCE = 3

# Model hyperparameters
LIWC_FEATURE_DIM = 17
HIDDEN_DIM = 128
DROPOUT = 0.25

# Emotion labels
EMOTION_LABELS = ['Positive', 'Negative', 'Neutral', 'Confident', 'Frustrated', 'Anxious']
EMOTION_LABEL_MAP = {'Positive': 0, 'Negative': 1, 'Neutral': 2, 'Confident': 3, 'Frustrated': 4, 'Anxious': 5}

# Formality labels
FORMALITY_LABELS = ['Informal', 'Formal']
FORMALITY_LABEL_MAP = {'Informal': 0, 'Formal': 1}
