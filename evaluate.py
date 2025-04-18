"""
Script for loading and evaluating a trained LSTNet model.
"""
import torch
import numpy as np
from pathlib import Path
import config
from src.data.data_processor import load_data, create_sequences, load_scaler
from src.visualization.plotter import plot_predictions, print_metrics
from OLD_FILES.LSTNet import LSTNet

def load_trained_model():
    # Load model
    model_path = Path(config.DATA_CONFIG['models_dir']) / 'lstnet_model.pth'
    checkpoint = torch.load(model_path)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTNet(
        num_features=len(checkpoint['feature_names']),
        device=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load scaler
    scaler, feature_names = load_scaler()
    
    return model, scaler, feature_names

def evaluate_model(model, scaler, feature_names, data_file=None):
    # Load and preprocess data
    print("\nLoading data...")
    dates, scaled_data, _, _ = load_data(data_file)
    
    # Create sequences
    print("Creating sequences...")
    X, y = create_sequences(scaled_data)
    
    # Generate predictions
    print("Generating predictions...")
    with torch.no_grad():
        predictions = model(X)
    
    # Convert to numpy arrays and inverse transform
    predictions = predictions.cpu().numpy()
    y = y.cpu().numpy()
    predictions = scaler.inverse_transform(predictions)
    y = scaler.inverse_transform(y)
    
    # Calculate metrics
    metrics = {}
    for i, feature in enumerate(feature_names):
        mse = np.mean((y[:, i] - predictions[:, i])**2)
        mae = np.mean(np.abs(y[:, i] - predictions[:, i]))
        mape = np.mean(np.abs((y[:, i] - predictions[:, i]) / y[:, i])) * 100
        
        metrics[feature] = {
            'MSE': mse,
            'MAE': mae,
            'MAPE': mape
        }
    
    # Visualize results
    print("\nVisualizing results...")
    plot_predictions(dates, y, predictions, feature_names, metrics)
    print_metrics(metrics)
    
    print("\nEvaluation complete! Results have been saved to the 'results' directory.")

def main():
    # Load trained model and scaler
    print("Loading trained model...")
    model, scaler, feature_names = load_trained_model()
    
    # Evaluate model
    evaluate_model(model, scaler, feature_names)

if __name__ == "__main__":
    main() 