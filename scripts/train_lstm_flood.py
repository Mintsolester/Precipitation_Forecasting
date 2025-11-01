"""
Train LSTM model for extreme precipitation/flood prediction
Predicts if precipitation will exceed flood threshold in next 7 days
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
DATA_FILE = "../data/processed/era5/ganges_farakka_era5_daily.csv"
OUTPUT_DIR = "../runs/lstm_flood_prediction"
SEQ_LENGTH = 365  # Use 1 year of history
FORECAST_HORIZON = 7  # Predict 7 days ahead
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3

# Flood threshold (95th percentile of precipitation)
FLOOD_PERCENTILE = 95

class FloodDataset(Dataset):
    """Dataset for flood prediction from ERA5 climate data"""
    
    def __init__(self, data, seq_length, forecast_horizon):
        self.data = data
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        
    def __len__(self):
        return len(self.data) - self.seq_length - self.forecast_horizon
    
    def __getitem__(self, idx):
        # Input: seq_length days of climate data
        x = self.data[idx:idx + self.seq_length, :-1]  # All features except target
        
        # Target: max precipitation in next forecast_horizon days
        future_precip = self.data[idx + self.seq_length:idx + self.seq_length + self.forecast_horizon, -1]
        y = np.max(future_precip)  # Maximum precipitation in forecast window
        
        return torch.FloatTensor(x), torch.FloatTensor([y])


class LSTMFloodModel(nn.Module):
    """LSTM model for flood prediction"""
    
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMFloodModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


def load_and_preprocess_data(data_file):
    """Load and preprocess ERA5 data"""
    
    print("=" * 70)
    print("📂 Loading Data")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv(data_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Features: {list(df.columns)}")
    
    # Select features for training
    feature_cols = ['total_precipitation', 'temperature_2m', 'soil_moisture_layer1', 'solar_radiation']
    
    # Add target column (precipitation for prediction)
    df['target_precipitation'] = df['total_precipitation'].copy()
    
    # Calculate flood threshold
    flood_threshold = np.percentile(df['total_precipitation'], FLOOD_PERCENTILE)
    print(f"\n🌊 Flood threshold ({FLOOD_PERCENTILE}th percentile): {flood_threshold:.2f} mm/day")
    
    # Extract features
    feature_cols_with_target = feature_cols + ['target_precipitation']
    data = df[feature_cols_with_target].values
    
    print(f"\nData shape: {data.shape}")
    print(f"Features: {len(feature_cols)}")
    
    return data, df, flood_threshold, feature_cols


def create_dataloaders(data, seq_length, forecast_horizon, test_size=0.2, val_size=0.1):
    """Create train, validation, and test dataloaders"""
    
    print("\n" + "=" * 70)
    print("🔄 Creating Dataloaders")
    print("=" * 70)
    
    # Normalize data (fit on training data only)
    total_samples = len(data) - seq_length - forecast_horizon
    train_val_size = int((1 - test_size) * total_samples)
    
    # Split indices
    scaler = StandardScaler()
    train_data = data[:train_val_size + seq_length]
    scaler.fit(train_data)
    
    # Normalize all data
    data_normalized = scaler.transform(data)
    
    # Create datasets
    dataset = FloodDataset(data_normalized, seq_length, forecast_horizon)
    
    # Split into train, val, test
    train_val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, 
        [int((1-test_size) * len(dataset)), len(dataset) - int((1-test_size) * len(dataset))]
    )
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_val_dataset,
        [int((1-val_size) * len(train_val_dataset)), len(train_val_dataset) - int((1-val_size) * len(train_val_dataset))]
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    """Train the LSTM model"""
    
    print("\n" + "=" * 70)
    print("🚀 Training LSTM Model")
    print("=" * 70)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Path(OUTPUT_DIR) / 'best_model.pt')
    
    return train_losses, val_losses


def evaluate_model(model, test_loader, device, scaler, flood_threshold):
    """Evaluate model on test set"""
    
    print("\n" + "=" * 70)
    print("📊 Evaluating Model")
    print("=" * 70)
    
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())
    
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()
    
    # Denormalize predictions
    # Note: predictions are in normalized space, need to denormalize
    dummy = np.zeros((len(predictions), scaler.n_features_in_))
    dummy[:, -1] = predictions  # Put predictions in last column (target)
    predictions_denorm = scaler.inverse_transform(dummy)[:, -1]
    
    dummy[:, -1] = actuals
    actuals_denorm = scaler.inverse_transform(dummy)[:, -1]
    
    # Calculate metrics
    mse = np.mean((predictions_denorm - actuals_denorm) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions_denorm - actuals_denorm))
    
    # Flood detection metrics
    actual_floods = actuals_denorm > flood_threshold
    predicted_floods = predictions_denorm > flood_threshold
    
    true_positives = np.sum(actual_floods & predicted_floods)
    false_positives = np.sum(~actual_floods & predicted_floods)
    false_negatives = np.sum(actual_floods & ~predicted_floods)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📈 Regression Metrics:")
    print(f"   RMSE: {rmse:.2f} mm/day")
    print(f"   MAE:  {mae:.2f} mm/day")
    
    print(f"\n🌊 Flood Detection Metrics:")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall:    {recall:.3f}")
    print(f"   F1-Score:  {f1_score:.3f}")
    print(f"   Actual flood events: {np.sum(actual_floods)}")
    print(f"   Predicted flood events: {np.sum(predicted_floods)}")
    
    return predictions_denorm, actuals_denorm, rmse, mae, f1_score


def plot_results(train_losses, val_losses, predictions, actuals):
    """Plot training results"""
    
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Plot training curves
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot predictions vs actuals
        plt.subplot(1, 2, 2)
        plt.scatter(actuals, predictions, alpha=0.5)
        plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
        plt.xlabel('Actual Precipitation (mm/day)')
        plt.ylabel('Predicted Precipitation (mm/day)')
        plt.title('Predictions vs Actuals')
        plt.grid(True)
        
        plt.savefig(output_dir / 'training_results.png', dpi=300, bbox_inches='tight')
        print(f"\n💾 Saved plot: {output_dir / 'training_results.png'}")
    except Exception as e:
        print(f"\n⚠️  Could not save plot (matplotlib error): {e}")
        print("   Training data saved successfully, plot skipped.")


def main():
    """Main training pipeline"""
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    
    # Load and preprocess data
    data, df, flood_threshold, feature_cols = load_and_preprocess_data(DATA_FILE)
    
    # Create dataloaders
    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        data, SEQ_LENGTH, FORECAST_HORIZON
    )
    
    # Initialize model
    input_size = len(feature_cols)
    model = LSTMFloodModel(input_size, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    
    print(f"\n🧠 Model Architecture:")
    print(f"   Input size: {input_size}")
    print(f"   Hidden size: {HIDDEN_SIZE}")
    print(f"   Num layers: {NUM_LAYERS}")
    print(f"   Dropout: {DROPOUT}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, EPOCHS
    )
    
    # Load best model
    model.load_state_dict(torch.load(Path(OUTPUT_DIR) / 'best_model.pt'))
    
    # Evaluate
    predictions, actuals, rmse, mae, f1_score = evaluate_model(
        model, test_loader, device, scaler, flood_threshold
    )
    
    # Plot results
    plot_results(train_losses, val_losses, predictions, actuals)
    
    # Save scaler
    import pickle
    with open(Path(OUTPUT_DIR) / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    print(f"📁 Model saved: {OUTPUT_DIR}/best_model.pt")
    print(f"📁 Scaler saved: {OUTPUT_DIR}/scaler.pkl")
    print(f"📁 Results plot: {OUTPUT_DIR}/training_results.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
