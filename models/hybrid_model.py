"""
Hybrid RoBERTa + LIWC Model
===========================
Hybrid model that combines:
1. RoBERTa embeddings (768-dim)
2. LIWC-like lexicon features
3. MLP for classification
"""

import torch
import torch.nn as nn
from transformers import RobertaModel


class HybridRoBERTaLIWCModel(nn.Module):
    """
    Hybrid model that combines:
    1. RoBERTa embeddings (768-dim)
    2. LIWC-like lexicon features
    3. MLP for classification
    """
    
    def __init__(self, num_classes, liwc_feature_dim=17, hidden_dim=128, dropout=0.25):
        super(HybridRoBERTaLIWCModel, self).__init__()
        
        # RoBERTa encoder
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.roberta_dim = 768
        
        # Combined dimension
        combined_dim = self.roberta_dim + liwc_feature_dim
        
        # Small MLP: single hidden layer with ReLU + dropout, then output
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, input_ids, attention_mask, liwc_features):
        # Get RoBERTa embeddings
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = roberta_output.pooler_output  # [batch_size, 768]
        
        # Concatenate with LIWC features
        combined = torch.cat([pooled_output, liwc_features], dim=1)
        
        # Pass through MLP
        logits = self.classifier(combined)
        
        return logits
