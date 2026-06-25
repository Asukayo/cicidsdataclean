import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import json

from models.Fedformer_cls import Model

from dataprovider.provider_6_1_3 import load_data, split_data_chronologically, create_data_loaders
from config import CICIDS_WINDOW_SIZE, CICIDS_WINDOW_STEP
from units.trainer_valder import train_epoch, val_epoch, test_with_detailed_metrics


class Config:
    """FEDformer classification config for CICIDS2017 anomaly detection."""

    def __init__(self):
        # ---- Data ----
        self.seq_len = CICIDS_WINDOW_SIZE   # 窗口长度
        self.enc_in = 38                    # 特征数，load_data 后会自动更新

        # ---- FEDformer architecture ----
        self.version = 'Fourier'            # 'Fourier' or 'Wavelets'
        self.mode_select = 'random'         # frequency mode selection: 'random' / 'low'
        self.modes = 32                     # number of Fourier modes to keep
        self.d_model = 128                   # embedding dimension
        self.n_heads = 8                    # must be 8 (FourierBlock weights hardcoded to 8 heads)
        self.d_ff = 512                     # feed-forward hidden dim
        self.e_layers = 4                   # encoder layers
        self.dropout = 0.1
        self.activation = 'gelu'
        self.moving_avg = 25                # moving average kernel for decomposition
        self.embed = 'timeF'                # embedding type
        self.freq = 'h'                     # time feature frequency

        # Wavelet-specific (only used when version='Wavelets')
        self.L = 1
        self.base = 'legendre'

        # ---- Classification ----
        self.num_classes = 2
        self.pred_len = 38                  # kept for compatibility if needed

        # ---- Training ----
        self.epochs = 50
        self.batch_size = 128
        self.learning_rate = 1e-8

        self.weight_decay = 1e-4
        self.patience = 10

        # ---- Data split ----
        self.train_ratio = 0.6
        self.val_ratio = 0.2

        # ---- Misc ----
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'fedformer')


def train_model(configs):
    os.makedirs(configs.save_dir, exist_ok=True)

    # 1. Load data
    data_dir = "../cicids2017/selected_features"
    X, y, metadata = load_data(data_dir, CICIDS_WINDOW_SIZE, CICIDS_WINDOW_STEP)
    configs.enc_in = len(metadata['feature_names'])
    print(f"Feature count: {configs.enc_in}")

    # 2. Split
    train_data, val_data, test_data = split_data_chronologically(
        X, y, configs.train_ratio, configs.val_ratio
    )

    # 3. Data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader, scaler = create_data_loaders(
        train_data, val_data, test_data, configs.batch_size
    )

    # 4. Model
    print("\nCreating FEDformer classifier...")
    model = Model(configs).to(configs.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # 5. Optimizer & scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate, weight_decay=configs.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 6. Training loop
    print("\nTraining...")
    best_val_f1 = 0
    patience_counter = 0
    train_history = []

    for epoch in range(configs.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, configs.device)
        val_loss, val_acc, val_precision, val_recall, val_f1 = val_epoch(
            model, val_loader, criterion, configs.device
        )
        scheduler.step(val_loss)

        epoch_stats = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'lr': optimizer.param_groups[0]['lr']
        }
        train_history.append(epoch_stats)

        print(f"Epoch {epoch + 1:3d}/{configs.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'configs': configs,
                'scaler': scaler,
                'best_val_f1': best_val_f1,
                'feature_names': metadata['feature_names']
            }
            torch.save(checkpoint, os.path.join(configs.save_dir, 'best_model.pth'))
            print(f"  → New best F1: {best_val_f1:.4f}, model saved!")
        else:
            patience_counter += 1

        if patience_counter >= configs.patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    with open(os.path.join(configs.save_dir, 'train_history.json'), 'w') as f:
        json.dump(train_history, f, indent=2)

    # 7. Test
    print("\nTesting best model...")
    best_checkpoint = torch.load(os.path.join(configs.save_dir, 'best_model.pth'))
    model.load_state_dict(best_checkpoint['model_state_dict'])

    test_loss, test_acc, test_precision, test_recall, test_f1, test_pr_auc, test_cm = test_with_detailed_metrics(
        model, test_loader, criterion, configs.device
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS (FEDformer)")
    print("=" * 60)
    print(f"Test Accuracy:      {test_acc:.4f}")
    print(f"Test Precision:     {test_precision:.4f}")
    print(f"Test Recall:        {test_recall:.4f}")
    print(f"Test F1:            {test_f1:.4f}")
    print(f"Test PR-AUC:        {test_pr_auc:.4f}")
    print("\nConfusion Matrix:")
    print(test_cm)
    print(f"  TN: {test_cm[0, 0]}  FP: {test_cm[0, 1]}")
    print(f"  FN: {test_cm[1, 0]}  TP: {test_cm[1, 1]}")

    return model, train_history


if __name__ == "__main__":
    configs = Config()
    model, history = train_model(configs)
    print("\nTraining completed!")