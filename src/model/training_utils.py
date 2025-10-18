import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import os
import json
import torch.nn.functional as F
from sklearn.metrics import (
    balanced_accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================================
# 4. CLASS-BALANCED LOSS (CB LOSS)
# ============================================================================

class CBLoss(nn.Module):
    """
    Class-Balanced Loss based on effective number of samples.
    
    Paper: "Class-Balanced Loss Based on Effective Number of Samples"
    https://arxiv.org/abs/1901.05555
    
    Args:
        samples_per_class: [C] number of samples per class
        beta: reweighting parameter (0.9-0.999 typical)
        loss_type: 'focal' or 'ce'
        gamma: focal loss gamma (if focal)
    """
    def __init__(self, samples_per_class, beta=0.9999, loss_type='focal', gamma=2.0):
        super().__init__()
        
        # Compute effective number of samples
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / weights.sum() * len(weights)
        
        self.weights = torch.FloatTensor(weights)
        self.loss_type = loss_type
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C] logits
            targets: [B] class indices
        """
        weights = self.weights.to(inputs.device)
        
        if self.loss_type == 'ce':
            return F.cross_entropy(inputs, targets, weight=weights)
        elif self.loss_type == 'focal':
            ce_loss = F.cross_entropy(inputs, targets, weight=weights, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = (1 - pt) ** self.gamma * ce_loss
            return focal_loss.mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


# ============================================================================
# 5. MIXUP AUGMENTATION
# ============================================================================

def mixup_data(x, y, alpha=0.2):
    """
    Mixup augmentation for ECG signals.
    
    Mixes pairs of examples to create synthetic training data.
    Works well for imbalanced datasets.
    
    Args:
        x: [B, 2, 256] input data
        y: [B] labels
        alpha: mixup beta distribution parameter
        
    Returns:
        mixed_x, mixed_y, lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss function for mixup."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss with optional class weighting.
    
    Args:
        alpha (float, list, torch.Tensor): Class weights or scalar.
            - If float, applies the same weight to all classes.
            - If list or tensor, length must equal num_classes.
        gamma (float): Focusing parameter (>0 focuses more on hard examples).
    """
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        if isinstance(alpha, (list, torch.Tensor)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = alpha  # scalar or None
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        inputs: [batch_size, num_classes] (logits)
        targets: [batch_size, num_clases] (One hot encoded labels)
        """
        device = inputs.device
        log_probs = F.log_softmax(inputs, dim=1)

        # Gather probabilities and log-probabilities of the true class
        pt = (torch.exp(log_probs) * targets).sum(dim=1)
        log_pt = (log_probs * targets).sum(dim=1)

        # Compute alpha weighting per class
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = (self.alpha.to(device)* targets).sum(dim=1)
            else:
                alpha_t = torch.tensor(self.alpha, dtype=torch.float32, device=device)
        else:
            alpha_t = 1.0

        # Focal Loss
        loss = -alpha_t * (1 - pt) ** self.gamma * log_pt

        return loss.mean()


class OneHotCrossEntropyLoss(nn.Module):
    """
    Computes Categorical Cross-Entropy Loss where targets are one-hot encoded.
    Always uses 'mean' reduction.
    """
    def __init__(self):
        super().__init__()


    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Logits (raw outputs) from the model. Shape [B, C].
            targets (torch.Tensor): One-hot encoded labels. Shape [B, C].
        """
            
        # Compute log probabilities
        log_probs = F.log_softmax(inputs, dim=1)

        # Compute negative log-likelihood per sample using the one-hot targets
        # Loss = - sum(y_i * log(p_i))
        loss_per_sample = -(log_probs * targets).sum(dim=1)

        # 4. Enforce mean reduction
        return loss_per_sample.mean()
        

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6):
    """Cosine learning rate schedule with warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return max(min_lr / optimizer.defaults['lr'], cosine_decay)
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=10, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def calculate_metrics(all_preds, all_targets, all_probs=None, num_classes=None):
    """
    Calculate classification metrics that handle class imbalance.
    
    Args:
        all_preds: Predicted class labels
        all_targets: True class labels
        all_probs: Predicted probabilities (optional, for AUC)
        num_classes: Number of classes (required for AUC)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Balanced Accuracy - accounts for class imbalance
    metrics['balanced_accuracy'] = balanced_accuracy_score(all_targets, all_preds)
    
    # Macro-averaged metrics - treat all classes equally
    metrics['macro_f1'] = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    metrics['macro_precision'] = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    metrics['macro_recall'] = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    
    # Weighted metrics - account for class distribution
    metrics['weighted_f1'] = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    # Regular accuracy for reference
    metrics['accuracy'] = (all_preds == all_targets).mean()
    
    # Per-class metrics
    per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)
    for i, score in enumerate(per_class_f1):
        metrics[f'f1_class_{i}'] = score
    
    # AUC scores if probabilities provided
    if all_probs is not None and num_classes is not None:
        try:
            if num_classes == 2:
                # Binary classification
                metrics['auc_roc'] = roc_auc_score(all_targets, all_probs[:, 1])
            else:
                # Multi-class (one-vs-rest, macro average)
                metrics['auc_roc_macro'] = roc_auc_score(
                    all_targets, all_probs, 
                    multi_class='ovr', 
                    average='macro'
                )
        except ValueError:
            # Handle cases where not all classes are present
            pass
    
    # Confusion matrix for deeper analysis
    cm = confusion_matrix(all_targets, all_preds)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def train_epoch(model, train_loader, optimizer, scheduler, device, mask_ratio, epoch, criterion=None, strategy=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    # For classification metrics
    all_preds = []
    all_targets = []
    all_probs = []
    
    with tqdm(
            total=len(train_loader), desc=f"Epoch {epoch} [Train]", unit="batch"
        ) as pbar:
        for batch in train_loader:

            if criterion is None:  # SSL
                batch = batch.to(device)
                output = model(batch, mask_ratio=mask_ratio)
                loss = output['loss']
            else:  # Classification
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Apply mixup if specified
                if strategy == 'mixup':
                    inputs, y_a, y_b, lam = mixup_data(inputs, targets, alpha=0.2)
                    outputs = model(inputs)
                    loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
                    # Note: Skip metrics calculation for mixup batches
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Collect predictions for metrics (only when not using mixup)
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)
                    targets = torch.argmax(targets, dim=1)
                    
                    all_preds.append(preds.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
                    all_probs.append(probs.detach().cpu().numpy())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })
            pbar.update(1)
    
    avg_loss = total_loss / num_batches
    
    # Calculate training metrics for classification
    train_metrics = {}
    if criterion is not None and len(all_preds) > 0 and strategy != 'mixup':
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_probs = np.concatenate(all_probs)
        num_classes = all_probs.shape[1]
        
        train_metrics = calculate_metrics(all_preds, all_targets, all_probs, num_classes)
    
    return avg_loss, train_metrics


@torch.no_grad()
def validate(model, val_loader, device, mask_ratio, epoch, criterion=None):
    """Validate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    # For classification metrics
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        with tqdm(
            total=len(val_loader), desc=f"Epoch {epoch} [Val]", unit="batch"
        ) as pbar:
            for batch in val_loader:
                if criterion is None:  # SSL
                    batch = batch.to(device)
                    output = model(batch, mask_ratio=mask_ratio)
                    loss = output['loss']
                else:  # Classification
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Collect predictions for metrics
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)
                    targets = torch.argmax(targets, dim=1)

                    all_preds.append(preds.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
                    all_probs.append(probs.cpu().numpy())
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                pbar.update(1)
    
    avg_loss = total_loss / num_batches
    
    # Calculate validation metrics for classification
    val_metrics = {}
    if criterion is not None and len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_probs = np.concatenate(all_probs)
        num_classes = all_probs.shape[1]
        
        val_metrics = calculate_metrics(all_preds, all_targets, all_probs, num_classes)
    
    return avg_loss, val_metrics


def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, path, train_metrics=None, val_metrics=None):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, scheduler, path, device):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return (checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss'],
            checkpoint.get('train_metrics'), checkpoint.get('val_metrics'))


def train(
    model,
    train_dataset,
    val_dataset,
    sampler=None,
    num_epochs=100,
    batch_size=64,
    learning_rate=1e-3,
    weight_decay=0.05,
    mask_ratio=0.60,
    warmup_epochs=10,
    device='cuda',
    save_dir='checkpoints',
    patience=15,
    num_workers=4,
    criterion=None,
    strategy=None,
    monitor_metric='macro_f1'):  # New parameter
    """
    Complete training loop for ECG MAE.
    
    Args:
        model: TransformerAutoencoder instance
        train_dataset: Training dataset
        val_dataset: Validation dataset
        sampler: Optional sampler for DataLoader (e.g. balanced)
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Peak learning rate (after warmup)
        weight_decay: AdamW weight decay
        mask_ratio: Fraction of patches to mask
        warmup_epochs: Number of warmup epochs
        device: 'cuda' or 'cpu'
        save_dir: Directory to save checkpoints
        patience: Early stopping patience
        num_workers: DataLoader workers
        criterion: Loss function for classification (None for SSL)
        strategy: Augmentation strategy ('mixup' or None)
        monitor_metric: Metric to monitor for best model ('balanced_accuracy', 'macro_f1', etc.)
    """
    # Setup
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if sampler is not None:
        train_loader = DataLoader(
            train_dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Optimizer
    param_groups = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if p.requires_grad and not any(x in n for x in ['bias', 'norm'])],
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if p.requires_grad and any(x in n for x in ['bias', 'norm'])],
            'weight_decay': 0.0
        }
    ]
    optimizer = optim.AdamW(param_groups, lr=learning_rate, betas=(0.9, 0.95))
    
    # Learning rate scheduler
    num_training_steps = num_epochs * len(train_loader)
    num_warmup_steps = warmup_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr=learning_rate * 0.01
    )
    
    # Early stopping - use max mode for metrics, min mode for loss
    if criterion is not None:
        early_stopping = EarlyStopping(patience=patience, min_delta=1e-4, mode='max')
    else:
        early_stopping = EarlyStopping(patience=patience, min_delta=1e-4, mode='min')
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': [],
        'train_metrics': [],
        'val_metrics': []
    }
    
    best_val_metric = -float('inf') if criterion is not None else float('inf')
    best_val_loss = float('inf')
    
    print(f"{'='*60}")
    print(f"Training Configuration:")
    print(f"  Device: {device}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Mask ratio: {mask_ratio}")
    print(f"  Warmup epochs: {warmup_epochs}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    if criterion is not None:
        print(f"  Monitor metric: {monitor_metric}")
    print(f"{'='*60}\n")
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, 
            device, mask_ratio, epoch, criterion, strategy
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, device, mask_ratio, epoch,
            criterion
        )
        
        # Track history
        current_lr = scheduler.get_last_lr()[0]
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(current_lr)
        history['train_metrics'].append(train_metrics)
        history['val_metrics'].append(val_metrics)
        
        # Print epoch summary
        print(f"\nEpoch {epoch:3d}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
        
        # Print classification metrics if available
        if criterion is not None and val_metrics:
            print(f"  Validation Metrics:")
            print(f"    Balanced Accuracy: {val_metrics.get('balanced_accuracy', 0):.4f}")
            print(f"    Macro F1:          {val_metrics.get('macro_f1', 0):.4f}")
            print(f"    Weighted F1:       {val_metrics.get('weighted_f1', 0):.4f}")
            print(f"    Accuracy:          {val_metrics.get('accuracy', 0):.4f}")
            if 'auc_roc' in val_metrics:
                print(f"    AUC-ROC:           {val_metrics['auc_roc']:.4f}")
            elif 'auc_roc_macro' in val_metrics:
                print(f"    AUC-ROC (macro):   {val_metrics['auc_roc_macro']:.4f}")
        
        # Determine whether to save based on criterion type
        save_best = False
        if criterion is None:
            # SSL: use validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_best = True
        else:
            # Classification: use specified metric
            current_metric = val_metrics.get(monitor_metric, -float('inf'))
            if current_metric > best_val_metric:
                best_val_metric = current_metric
                save_best = True
        
        # Save best model
        if save_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_loss, val_loss,
                os.path.join(save_dir, 'best_model.pt'),
                train_metrics, val_metrics
            )
            if criterion is None:
                print(f"  → Saved best model (val_loss: {val_loss:.4f})")
            else:
                print(f"  → Saved best model ({monitor_metric}: {best_val_metric:.4f})")
        
        # Save regular checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_loss, val_loss,
                os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt'),
                train_metrics, val_metrics
            )
        
        # Early stopping check
        check_value = val_metrics.get(monitor_metric, -float('inf')) if criterion is not None else val_loss
        if early_stopping(check_value):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
    
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, epoch,
        train_loss, val_loss,
        os.path.join(save_dir, 'final_model.pt'),
        train_metrics, val_metrics
    )
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    if criterion is None:
        print(f"Best validation loss: {best_val_loss:.4f}")
    else:
        print(f"Best {monitor_metric}: {best_val_metric:.4f}")
    print(f"Final model saved to: {save_dir}/final_model.pt")
    print(f"{'='*60}\n")
    
    # Save training history
    exp_name = "history.json"
    with open(os.path.join(save_dir, exp_name), 'w') as f:
        json.dump(history, f, indent=4, default=lambda x: x if isinstance(x, (int, float, str, list)) else str(x))
    
    return history