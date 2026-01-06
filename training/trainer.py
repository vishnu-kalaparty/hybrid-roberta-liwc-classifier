"""
Training and evaluation functions.
"""

import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


def train_epoch(model, dataloader, optimizer, criterion, device, gradient_accumulation_steps=1):
    """Train for one epoch with optional gradient accumulation."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        liwc_features = batch['liwc_features'].to(device)
        labels = batch['label'].to(device)
        
        logits = model(input_ids, attention_mask, liwc_features)
        loss = criterion(logits, labels)
        loss = loss / gradient_accumulation_steps  # Scale loss for accumulation
        
        loss.backward()
        
        # Update weights every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps  # Unscale for logging
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    # Handle remaining gradients if any
    if len(dataloader) % gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, accuracy, f1


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            liwc_features = batch['liwc_features'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask, liwc_features)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, accuracy, f1, all_preds, all_labels


def train_model(model, train_loader, val_loader, optimizer, criterion, device, 
                num_epochs=5, patience=3, gradient_accumulation_steps=1):
    """Full training loop with early stopping and gradient accumulation."""
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print('='*50)
        
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, criterion, 
                                                      device, gradient_accumulation_steps)
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Macro-F1: {train_f1:.4f}")
        
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        print(f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Macro-F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    return model
