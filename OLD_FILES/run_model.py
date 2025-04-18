import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
from OLD_FILES.LSTNet import LSTNet
from datetime import datetime

# Create results directory
os.makedirs('results', exist_ok=True)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_and_preprocess_data(data_file='multivariate_unemployment_LSTNet.csv'):
    """Load and preprocess the unemployment data"""
    data = pd.read_csv(f'data/{data_file}')
    
    # Handle different file formats
    if 'date' in data.columns:
        # Multivariate format
        dates = pd.to_datetime(data['date'])
        features = data.drop('date', axis=1)
        feature_names = features.columns.tolist()
    else:
        # Single series format
        dates = pd.to_datetime(data['Year'].astype(str) + '-' + data['Period'].str[1:])
        features = data[['Value']]
        feature_names = ['Unemployment Rate']
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)
    
    return dates, scaled_data, scaler, feature_names

def create_sequences(data, seq_length):
    """Create sequences for time series prediction"""
    X = np.zeros((len(data) - seq_length, seq_length, data.shape[1]))
    y = np.zeros((len(data) - seq_length, data.shape[1]))
    
    for i in range(len(data) - seq_length):
        X[i] = data[i:(i + seq_length)]
        y[i] = data[i + seq_length]
    
    return torch.FloatTensor(X).to(device), torch.FloatTensor(y).to(device)

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """Train the LSTNet model"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
        
        train_losses.append(total_loss / len(X_train))
        val_losses.append(val_loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
    
    return train_losses, val_losses

def calculate_metrics(y_true, y_pred, feature_names):
    """Calculate various metrics for each feature"""
    metrics = {}
    for i, feature in enumerate(feature_names):
        mse = np.mean((y_true[:, i] - y_pred[:, i])**2)
        mae = np.mean(np.abs(y_true[:, i] - y_pred[:, i]))
        mape = np.mean(np.abs((y_true[:, i] - y_pred[:, i]) / y_true[:, i])) * 100
        
        metrics[feature] = {
            'MSE': mse,
            'MAE': mae,
            'MAPE': mape
        }
    
    return metrics

def plot_predictions(model, X_test, y_test, dates, scaler, feature_names):
    """Plot actual vs predicted values"""
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
    
    # Inverse transform the predictions and actual values
    predictions = scaler.inverse_transform(predictions.cpu().numpy())
    actual = scaler.inverse_transform(y_test.cpu().numpy())
    
    # Calculate metrics
    metrics = calculate_metrics(actual, predictions, feature_names)
    
    # Plot each feature
    for i, feature in enumerate(feature_names):
        plt.figure(figsize=(12, 6))
        plt.plot(dates[-len(actual):], actual[:, i], label='Actual', color='blue')
        plt.plot(dates[-len(predictions):], predictions[:, i], label='Predicted', color='red', linestyle='--')
        plt.title(f'{feature} - Actual vs Predicted\nMSE: {metrics[feature]["MSE"]:.4f}, MAE: {metrics[feature]["MAE"]:.4f}, MAPE: {metrics[feature]["MAPE"]:.2f}%')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'results/{feature}_prediction.png')
        plt.close()
    
    return metrics

def plot_loss_curves(train_losses, val_losses):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('results/loss_curves.png')
    plt.close()

def main():
    # Load and preprocess data
    print("Loading data...")
    dates, scaled_data, scaler, feature_names = load_and_preprocess_data()
    print(f"Features: {feature_names}")
    
    # Create sequences
    seq_length = 12  # Using 12 months as sequence length
    X, y = create_sequences(scaled_data, seq_length)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)
    
    # Initialize model with correct number of features
    model = LSTNet(num_features=len(feature_names), device=device)
    
    # Train model
    print("Starting training...")
    train_losses, val_losses = train_model(model, X_train, y_train, X_val, y_val)
    
    # Plot loss curves
    plot_loss_curves(train_losses, val_losses)
    
    # Plot predictions and get metrics
    print("Generating prediction plots...")
    metrics = plot_predictions(model, X_test, y_test, dates, scaler, feature_names)
    
    # Print final metrics
    print("\nFinal Metrics:")
    for feature, feature_metrics in metrics.items():
        print(f"\n{feature}:")
        print(f"  MSE: {feature_metrics['MSE']:.4f}")
        print(f"  MAE: {feature_metrics['MAE']:.4f}")
        print(f"  MAPE: {feature_metrics['MAPE']:.2f}%")
    
    print("\nPlots have been saved to the 'results' directory.")

if __name__ == "__main__":
    main() 