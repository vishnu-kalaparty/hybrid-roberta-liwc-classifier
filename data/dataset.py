"""
Dataset class for text classification.
"""

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Dataset class for text classification."""
    
    def __init__(self, texts, labels, tokenizer, liwc_extractor, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.liwc_extractor = liwc_extractor
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize for RoBERTa
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Extract LIWC-like features
        liwc_features = torch.tensor(self.liwc_extractor.extract_features(text), dtype=torch.float)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'liwc_features': liwc_features,
            'label': torch.tensor(label, dtype=torch.long)
        }
