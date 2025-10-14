import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import os
import json
import torch.nn.functional as F



import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
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
        probs = torch.exp(log_probs)

        # Gather probabilities and log-probabilities of the true class
        pt = (probs * targets).sum(dim=1)
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


def train_epoch(model, train_loader, optimizer, scheduler, device, mask_ratio, epoch, criterion=None, strategy=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    with tqdm(
            total=len(train_loader), desc=f"Epoch {epoch} [Train]", unit="batch"
        ) as pbar:
        for batch in train_loader:

            if criterion is None: # SSL
                batch = batch.to(device)  # [B, 2, 256]
                # Forward pass
                output = model(batch, mask_ratio=mask_ratio)
                loss = output['loss']
            else: # Classification
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                # Apply mixup if specified
                if strategy == 'mixup':
                    inputs, y_a, y_b, lam = mixup_data(inputs, targets, alpha=0.2)
                    outputs = model(inputs)
                    loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (important for transformer stability)
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
    return avg_loss


@torch.no_grad()
def validate(model, val_loader, device, mask_ratio, epoch, criterion=None):
    """Validate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        with tqdm(
            total=len(val_loader), desc=f"Epoch {epoch} [Val]", unit="batch"
        ) as pbar:
            for batch in val_loader:
                if criterion is None: # SSL
                    batch = batch.to(device)  # [B, 2, 256]
                    # Forward pass
                    output = model(batch, mask_ratio=mask_ratio)
                    loss = output['loss']
                else: # Classification
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                pbar.update(1)
    
    avg_loss = total_loss / num_batches
    return avg_loss


def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, scheduler, path, device):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False )
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']


def train(
    model,
    train_dataset,
    val_dataset,
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
    strategy=None):
    """
    Complete training loop for ECG MAE.
    
    Args:
        model: TransformerAutoencoder instance
        train_dataset: Training dataset
        val_dataset: Validation dataset
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
    """
    # Setup
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # For stable batch norm
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Optimizer (AdamW with weight decay, excluding biases and norms)
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
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=1e-4, mode='min')
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
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
    print(f"{'='*60}\n")
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, 
            device, mask_ratio, epoch, criterion, strategy
        )
        
        # Validate
        val_loss = validate(
            model, val_loader, device, mask_ratio, epoch,
            criterion
        )
        
        # Track history
        current_lr = scheduler.get_last_lr()[0]
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(current_lr)
        
        # Print epoch summary
        print(f"Epoch {epoch:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"LR: {current_lr:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_loss, val_loss,
                os.path.join(save_dir, 'best_model.pt')
            )
            print(f"  → Saved best model (val_loss: {val_loss:.4f})")
        
        # Save regular checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_loss, val_loss,
                os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
            )
        
        # Early stopping check
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
    
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, epoch,
        train_loss, val_loss,
        os.path.join(save_dir, 'final_model.pt')
    )
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final model saved to: {save_dir}/final_model.pt")
    print(f"{'='*60}\n")
    
    # Save training history
    exp_name = "history.json"
    with open(os.path.join(save_dir, exp_name), 'w') as f:
        json.dump(history, f, indent=4)