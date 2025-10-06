import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import os
import json



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


def train_epoch(model, train_loader, optimizer, scheduler, device, mask_ratio, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    with tqdm(
            total=len(train_loader), desc=f"Epoch {epoch} [Train]", unit="batch"
        ) as pbar:
        for batch in train_loader:
            batch = batch.to(device)  # [B, 2, 256]
            
            # Forward pass
            output = model(batch, mask_ratio=mask_ratio)
            loss = output['loss']
            
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
def validate(model, val_loader, device, mask_ratio, epoch):
    """Validate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        with tqdm(
            total=len(val_loader), desc=f"Epoch {epoch} [Val]", unit="batch"
        ) as pbar:
            for batch in val_loader:
                batch = batch.to(device)
                
                # Forward pass
                output = model(batch, mask_ratio=mask_ratio)
                loss = output['loss']
                
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


def train_mae(
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
):
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
            device, mask_ratio, epoch
        )
        
        # Validate
        val_loss = validate(
            model, val_loader, device, mask_ratio, epoch
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