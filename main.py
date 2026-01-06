"""
Hybrid Model: RoBERTa + LIWC-like Features + MLP
================================================
This implements the 3rd hybrid approach for the Emotion-Aware Academic Writing Assistant.

The model combines:
1. RoBERTa embeddings (contextual representations)
2. LIWC-like lexicon features (interpretable linguistic cues)
3. MLP classifier (lightweight fusion and classification)

Tasks:
- Emotion Detection: 6 classes (Positive, Negative, Neutral, Confident, Frustrated, Anxious)
- Formality Classification: 2 classes (Formal, Informal)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
from sklearn.utils import resample

from config import (
    device, DATA_DIR, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, NUM_EPOCHS,
    MAX_LENGTH, PATIENCE, LIWC_FEATURE_DIM, HIDDEN_DIM, DROPOUT,
    EMOTION_LABELS, FORMALITY_LABELS
)
from features import LIWCFeatureExtractor
from data import TextDataset, load_goemotions_data, load_formality_data
from models import HybridRoBERTaLIWCModel
from training import train_model, evaluate

# Print device info
print(f"Using device: {device}")


def main():
    print("="*60)
    print("Hybrid Model: RoBERTa + LIWC-like Features + MLP")
    print("Emotion-Aware Academic Writing Assistant")
    print("="*60)
    
    # Initialize components
    print("\nInitializing tokenizer and feature extractor...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    liwc_extractor = LIWCFeatureExtractor()
    
    
    # TASK 1: Emotion Detection
    
    print("\n" + "="*60)
    print("TASK 1: EMOTION DETECTION")
    print("="*60)
    
    # Load emotion data
    emotion_data = load_goemotions_data(DATA_DIR)
    
    # Use full dataset for training
    print(f"\nUsing full dataset: {len(emotion_data)} examples")
    
    # Split data
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        emotion_data['text'].tolist(),
        emotion_data['emotion_encoded'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=emotion_data['emotion_encoded']
    )
    
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42
    )
    
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    
    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, liwc_extractor, MAX_LENGTH)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, liwc_extractor, MAX_LENGTH)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, liwc_extractor, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True if device.type != 'cpu' else False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True if device.type != 'cpu' else False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True if device.type != 'cpu' else False)
    
    # Initialize model
    print("\nInitializing Hybrid Model for Emotion Detection...")
    emotion_model = HybridRoBERTaLIWCModel(
        num_classes=6,
        liwc_feature_dim=LIWC_FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT
    ).to(device)
    
    # Compile model for faster training (PyTorch 2.0+)
    try:
        emotion_model = torch.compile(emotion_model, mode='reduce-overhead')
        print("Model compiled for faster training")
    except:
        print("torch.compile not available, using standard model")
    
    # UNFREEZE RoBERTa - we need its contextual understanding!
    # Use different learning rates: lower for RoBERTa, higher for classifier
    roberta_params = list(emotion_model.roberta.parameters())
    classifier_params = list(emotion_model.classifier.parameters())
    
    # Optimized learning rates for larger batch size (slightly increased for faster convergence)
    optimizer = torch.optim.AdamW([
        {'params': roberta_params, 'lr': 1.2e-5, 'weight_decay': 0.01},  # Slightly higher for larger batch
        {'params': classifier_params, 'lr': 2.4e-4, 'weight_decay': 0.01}  # Slightly higher for larger batch
    ])
    
    # Calculate class weights to handle imbalance
    emotion_counts = emotion_data['emotion_label'].value_counts()
    total = len(emotion_data)
    class_weights = []
    for label in EMOTION_LABELS:
        count = emotion_counts.get(label, 1)
        weight = total / (6 * count)  # Inverse frequency weighting
        class_weights.append(weight)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class weights: {dict(zip(EMOTION_LABELS, class_weights.tolist()))}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Train
    print("\nTraining Emotion Detection Model...")
    print(f"Using gradient accumulation: {GRADIENT_ACCUMULATION_STEPS} steps (effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    emotion_model = train_model(emotion_model, train_loader, val_loader, 
                                optimizer, criterion, device, NUM_EPOCHS, PATIENCE, GRADIENT_ACCUMULATION_STEPS)
    
    # Final evaluation
    print("\n" + "="*60)
    print("EMOTION DETECTION - Final Test Results")
    print("="*60)
    
    test_loss, test_acc, test_f1, preds, labels = evaluate(emotion_model, test_loader, criterion, device)
    
    emotion_labels_all = EMOTION_LABELS
    unique_labels = sorted(set(labels) | set(preds))
    emotion_labels_present = [emotion_labels_all[i] for i in unique_labels]
    
    # Calculate metrics
    macro_precision = precision_score(labels, preds, average='macro', zero_division=0)
    macro_recall = recall_score(labels, preds, average='macro', zero_division=0)
    
    print(f"\nLoading data for task: emotion")
    print(f"Test samples: {len(labels)}")
    
    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Macro-F1: {test_f1:.4f}")
    print(f"Macro-Precision: {macro_precision:.4f}")
    print(f"Macro-Recall: {macro_recall:.4f}")
    
    # Per-class metrics table
    print("\n=== Per-Class Metrics ===")
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 65)
    
    for i, label_idx in enumerate(unique_labels):
        label_name = emotion_labels_all[label_idx]
        # Get metrics for this class
        true_pos = sum(1 for p, l in zip(preds, labels) if p == label_idx and l == label_idx)
        false_pos = sum(1 for p, l in zip(preds, labels) if p == label_idx and l != label_idx)
        false_neg = sum(1 for p, l in zip(preds, labels) if p != label_idx and l == label_idx)
        support = sum(1 for l in labels if l == label_idx)
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{label_name:<15} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<10}")
    
    # Bootstrap confidence interval
    print("\n=== Bootstrap Confidence Interval (95% CI) ===")
    n_bootstrap = 1000
    f1_scores = []
    for _ in range(n_bootstrap):
        indices = resample(range(len(labels)), replace=True, n_samples=len(labels))
        boot_labels = [labels[i] for i in indices]
        boot_preds = [preds[i] for i in indices]
        f1_scores.append(f1_score(boot_labels, boot_preds, average='macro', zero_division=0))
    
    ci_lower = np.percentile(f1_scores, 2.5)
    ci_upper = np.percentile(f1_scores, 97.5)
    print(f"Macro-F1: {test_f1:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(labels, preds, labels=unique_labels)
    print(pd.DataFrame(cm, index=emotion_labels_present, columns=emotion_labels_present))
    
    
    # TASK 2: Formality Classification
    
    print("\n" + "="*60)
    print("TASK 2: FORMALITY CLASSIFICATION")
    print("="*60)
    
    # Load formality data
    formality_data = load_formality_data(DATA_DIR)
    
    # Use full dataset for training
    print(f"\nUsing full dataset: {len(formality_data)} examples")
    
    # Split data
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        formality_data['text'].tolist(),
        formality_data['formality_encoded'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=formality_data['formality_encoded']
    )
    
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42
    )
    
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    
    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, liwc_extractor, MAX_LENGTH)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, liwc_extractor, MAX_LENGTH)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, liwc_extractor, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True if device.type != 'cpu' else False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True if device.type != 'cpu' else False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True if device.type != 'cpu' else False)
    
    # Initialize model
    print("\nInitializing Hybrid Model for Formality Classification...")
    formality_model = HybridRoBERTaLIWCModel(
        num_classes=2,
        liwc_feature_dim=LIWC_FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT
    ).to(device)
    
    # Compile model for faster training (PyTorch 2.0+)
    try:
        formality_model = torch.compile(formality_model, mode='reduce-overhead')
        print("Model compiled for faster training")
    except:
        print("torch.compile not available, using standard model")
    
    # UNFREEZE RoBERTa with different learning rates
    roberta_params = list(formality_model.roberta.parameters())
    classifier_params = list(formality_model.classifier.parameters())
    
    # Optimized learning rates for full dataset training
    optimizer = torch.optim.AdamW([
        {'params': roberta_params, 'lr': 1e-5, 'weight_decay': 0.01},
        {'params': classifier_params, 'lr': 2e-4, 'weight_decay': 0.01}
    ])
    
    # Class weights for formality
    formality_counts = formality_data['formality_label'].value_counts()
    total = len(formality_data)
    class_weights = []
    for label in FORMALITY_LABELS:
        count = formality_counts.get(label, 1)
        weight = total / (2 * count)
        class_weights.append(weight)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class weights: {dict(zip(FORMALITY_LABELS, class_weights.tolist()))}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Train
    print("\nTraining Formality Classification Model...")
    print(f"Using gradient accumulation: {GRADIENT_ACCUMULATION_STEPS} steps (effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    formality_model = train_model(formality_model, train_loader, val_loader,
                                  optimizer, criterion, device, NUM_EPOCHS, PATIENCE, GRADIENT_ACCUMULATION_STEPS)
    
    # Final evaluation
    print("\n" + "="*60)
    print("FORMALITY CLASSIFICATION - Final Test Results")
    print("="*60)
    
    test_loss, test_acc, test_f1, preds, labels = evaluate(formality_model, test_loader, criterion, device)
    
    formality_labels_all = FORMALITY_LABELS
    unique_labels = sorted(set(labels) | set(preds))
    formality_labels_present = [formality_labels_all[i] for i in unique_labels]
    
    # Calculate metrics
    macro_precision = precision_score(labels, preds, average='macro', zero_division=0)
    macro_recall = recall_score(labels, preds, average='macro', zero_division=0)
    
    print(f"\nLoading data for task: formality")
    print(f"Test samples: {len(labels)}")
    
    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Macro-F1: {test_f1:.4f}")
    print(f"Macro-Precision: {macro_precision:.4f}")
    print(f"Macro-Recall: {macro_recall:.4f}")
    
    # Per-class metrics table
    print("\n=== Per-Class Metrics ===")
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 65)
    
    for i, label_idx in enumerate(unique_labels):
        label_name = formality_labels_all[label_idx]
        true_pos = sum(1 for p, l in zip(preds, labels) if p == label_idx and l == label_idx)
        false_pos = sum(1 for p, l in zip(preds, labels) if p == label_idx and l != label_idx)
        false_neg = sum(1 for p, l in zip(preds, labels) if p != label_idx and l == label_idx)
        support = sum(1 for l in labels if l == label_idx)
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{label_name:<15} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<10}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(labels, preds, labels=unique_labels)
    print(pd.DataFrame(cm, index=formality_labels_present, columns=formality_labels_present))
    
    
    # Demo: Test on sample sentences
    
    print("\n" + "="*60)
    print("DEMO: Testing on Sample Sentences")
    print("="*60)
    
    demo_sentences = [
        "I was kinda frustrated with the results.",
        "The results indicate a significant improvement.",
        "I'm not sure this section is clear.",
        "This is absolutely brilliant work!",
        "yeah, this kinda worked",
        "The methodology demonstrates a robust approach to the problem.",
        "I'm worried this won't be accepted.",
        "ugh I'm so frustrated with this assignment, it's driving me crazy!",
        "omg I am really worried about the exam tomorrow"
    ]
    
    for sentence in demo_sentences:
        print(f"\nText: {sentence}")
        print("=" * 60)
        
        # Prepare input
        encoding = tokenizer(sentence, truncation=True, padding='max_length', 
                            max_length=MAX_LENGTH, return_tensors='pt')
        liwc_features = torch.tensor([liwc_extractor.extract_features(sentence)], dtype=torch.float)
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        liwc_features = liwc_features.to(device)
        
        # Predict emotion with probabilities
        emotion_model.eval()
        with torch.no_grad():
            emotion_logits = emotion_model(input_ids, attention_mask, liwc_features)
            emotion_probs = torch.softmax(emotion_logits, dim=1).squeeze().cpu().numpy()
            emotion_pred = torch.argmax(emotion_logits, dim=1).item()
        
        # Predict formality with probabilities
        formality_model.eval()
        with torch.no_grad():
            formality_logits = formality_model(input_ids, attention_mask, liwc_features)
            formality_probs = torch.softmax(formality_logits, dim=1).squeeze().cpu().numpy()
            formality_pred = torch.argmax(formality_logits, dim=1).item()
        
        emotion_name = emotion_labels_all[emotion_pred]
        formality_name = formality_labels_all[formality_pred]
        
        # Print emotion results
        print("\nEMOTION:")
        print(f"  Prediction: {emotion_name.lower()}")
        print("  Probabilities:")
        # Sort by probability (descending)
        emotion_sorted = sorted(zip(emotion_labels_all, emotion_probs), key=lambda x: x[1], reverse=True)
        for label, prob in emotion_sorted:
            print(f"    {label.lower()}: {prob:.4f}")
        
        # Print formality results
        print("\nFORMALITY:")
        print(f"  Prediction: {formality_name.lower()}")
        print("  Probabilities:")
        formality_sorted = sorted(zip(formality_labels_all, formality_probs), key=lambda x: x[1], reverse=True)
        for label, prob in formality_sorted:
            print(f"    {label.lower()}: {prob:.4f}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("\nThe hybrid model combines RoBERTa's contextual understanding")
    print("with LIWC-like lexicon features for interpretable predictions.")


if __name__ == "__main__":
    main()
