import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
from LSTNet import LSTNet

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_and_preprocess_data():
    """Load and preprocess the multivariate unemployment data"""
    data = pd.read_csv('data/multivariate_unemployment_LSTNet.csv')
    dates = pd.to_datetime(data['date'])
    data = data.drop('date', axis=1)
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    return dates, scaled_data, scaler

def create_sequences(data, seq_length):
    """Create sequences for time series prediction"""
    # Pre-allocate numpy arrays
    X = np.zeros((len(data) - seq_length, seq_length, data.shape[1]))
    y = np.zeros((len(data) - seq_length, data.shape[1]))
    
    # Fill arrays
    for i in range(len(data) - seq_length):
        X[i] = data[i:(i + seq_length)]
        y[i] = data[i + seq_length]
    
    # Convert to tensors in one go
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

def plot_predictions(model, X_test, y_test, dates, scaler, feature_names):
    """Plot actual vs predicted values"""
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
    
    # Inverse transform the predictions and actual values
    predictions = scaler.inverse_transform(predictions.cpu().numpy())
    actual = scaler.inverse_transform(y_test.cpu().numpy())
    
    # Plot each feature
    for i, feature in enumerate(feature_names):
        plt.figure(figsize=(12, 6))
        plt.plot(dates[-len(actual):], actual[:, i], label='Actual', color='blue')
        plt.plot(dates[-len(predictions):], predictions[:, i], label='Predicted', color='red', linestyle='--')
        plt.title(f'{feature} - Actual vs Predicted')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'results/{feature}_prediction.png')
        plt.close()

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
    dates, scaled_data, scaler = load_and_preprocess_data()
    feature_names = ['Unemployment Rate', 'Labor Force Participation Rate', 
                    'Employment-Population (Men)', 'Employment-Population (Women)',
                    'U-6 Unemployment Rate']
    
    # Create sequences
    seq_length = 12  # Using 12 months as sequence length
    X, y = create_sequences(scaled_data, seq_length)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)
    
    # Initialize model
    model = LSTNet().to(device)
    
    # Train model
    print("Starting training...")
    train_losses, val_losses = train_model(model, X_train, y_train, X_val, y_val)
    
    # Plot loss curves
    plot_loss_curves(train_losses, val_losses)
    
    # Plot predictions
    print("Generating prediction plots...")
    plot_predictions(model, X_test, y_test, dates, scaler, feature_names)
    
    # Calculate and print final metrics
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        mse = nn.MSELoss()(predictions, y_test)
        mae = nn.L1Loss()(predictions, y_test)
    
    print(f"\nFinal Metrics:")
    print(f"Test MSE: {mse.item():.4f}")
    print(f"Test MAE: {mae.item():.4f}")
    print("\nPlots have been saved to the 'results' directory.")

if __name__ == "__main__":
    main() 