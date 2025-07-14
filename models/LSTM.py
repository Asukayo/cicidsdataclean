import numpy as np
import pickle
import os
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import warnings
from lstmModel import LSTMModel

from config import CICIDS_WINDOW_SIZE,CICIDS_WINDOW_STEP


warnings.filterwarnings('ignore')



class LSTMDetector:
    def __init__(self, num_features=38, hidden_size=64, device=None):
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.model = None
        self.scaler = StandardScaler()
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def load_data(self, data_dir, window_size=CICIDS_WINDOW_SIZE, step_size=CICIDS_WINDOW_STEP):
        print("Loading data...")

        X_file = os.path.join(data_dir, f'selected_X_w{window_size}_s{step_size}.npy')
        y_file = os.path.join(data_dir, f'selected_y_w{window_size}_s{step_size}.npy')
        metadata_file = os.path.join(data_dir, f'selected_metadata_w{window_size}_s{step_size}.pkl')

        X = np.load(X_file)
        y = np.load(y_file)

        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)

        window_metadata = metadata['window_metadata']
        y_binary = np.array([w['is_malicious'] for w in window_metadata])

        print(f"X shape: {X.shape}, y shape: {y_binary.shape}")
        print(f"Normal: {np.sum(y_binary == 0)}, Anomalous: {np.sum(y_binary == 1)}")

        return X, y_binary

    def preprocess_data(self, X):
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.fit_transform(X_reshaped)
        return X_scaled.reshape(original_shape)

    def create_data_loaders(self, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
        datasets = []
        for X, y in [(X_train, y_train), (X_val, y_val), (X_test, y_test)]:
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.LongTensor(y)
            datasets.append(TensorDataset(X_tensor, y_tensor))

        train_loader = DataLoader(datasets[0], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(datasets[1], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(datasets[2], batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def build_model(self):
        self.model = LSTMModel(self.num_features, self.hidden_size).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model built with {total_params:,} parameters")
        return self.model

    def train_epoch(self, data_loader, criterion, optimizer=None):
        is_training = optimizer is not None
        self.model.train() if is_training else self.model.eval()

        total_loss = 0
        correct = 0
        total = 0

        with torch.set_grad_enabled(is_training):
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                if is_training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        return total_loss / len(data_loader), correct / total

    def train(self, train_loader, val_loader, epochs=50, lr=0.001, patience=10):
        print("Training model...")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr,weight_decay=1e-2)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)

            # Validation
            val_loss, val_acc = self.train_epoch(val_loader, criterion)

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        print("Training completed!")

    def evaluate(self, test_loader):
        print("Evaluating model...")

        self.model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_X)
                _, predictions = torch.max(outputs, 1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)

        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, target_names=['Normal', 'Anomalous']))

        return accuracy, f1

    def plot_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        ax1.grid(True)

        # Accuracy
        ax2.plot(self.history['train_acc'], label='Train Accuracy')
        ax2.plot(self.history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    data_dir = "../cicids2017/selected_features"

    # Initialize detector
    detector = LSTMDetector(num_features=38, hidden_size=64)

    # Load and preprocess data
    X, y = detector.load_data(data_dir)
    X_scaled = detector.preprocess_data(X)

    # Split data: 60% train, 20% val, 20% test
    # 修改后 - 避免数据泄露
    n_samples = len(X_scaled)
    train_end = int(n_samples * 0.6)
    val_end = int(n_samples * 0.8)

    X_train, y_train = X_scaled[:train_end], y[:train_end]
    X_val, y_val = X_scaled[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X_scaled[val_end:], y[val_end:]
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Create data loaders
    train_loader, val_loader, test_loader = detector.create_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64
    )

    # Build and train model
    detector.build_model()
    detector.train(train_loader, val_loader, epochs=50, lr=0.000001, patience=10)

    # Evaluate
    accuracy, f1 = detector.evaluate(test_loader)

    # Plot results
    detector.plot_history()

    print(f"\nFinal Results - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")


if __name__ == "__main__":
    main()