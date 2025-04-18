"""
Data processing module for handling unemployment data.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from pathlib import Path
import config

def load_data(data_file=None):
    """
    Load and preprocess the unemployment data.
    
    Args:
        data_file (str, optional): Name of the data file to load. If None, uses default.
        
    Returns:
        tuple: (dates, scaled_data, scaler, feature_names)
    """
    if data_file is None:
        data_file = config.DATA_CONFIG['default_data_file']
    
    # Load the data
    data_path = Path(config.DATA_CONFIG['data_dir']) / data_file
    data = pd.read_csv(data_path)
    
    # Handle different file formats
    if 'date' in data.columns:
        # Multivariate format (5 features)
        dates = pd.to_datetime(data['date'])
        features = data.drop('date', axis=1)
        feature_names = features.columns.tolist()
    else:
        # Single series format (1 feature)
        dates = pd.to_datetime(data['Year'].astype(str) + '-' + data['Period'].str[1:])
        features = data[['Value']]
        feature_names = ['Unemployment Rate']
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)
    
    return dates, scaled_data, scaler, feature_names

def create_sequences(data, seq_length=None):
    """
    Create sequences for time series prediction.
    
    Args:
        data (np.ndarray): Scaled data array
        seq_length (int, optional): Length of sequences. If None, uses config value.
        
    Returns:
        tuple: (X, y) tensors for input and target sequences
    """
    if seq_length is None:
        seq_length = config.TRAINING_CONFIG['sequence_length']
    
    # Create sequences
    X = np.zeros((len(data) - seq_length, seq_length, data.shape[1]))
    y = np.zeros((len(data) - seq_length, data.shape[1]))
    
    for i in range(len(data) - seq_length):
        X[i] = data[i:(i + seq_length)]
        y[i] = data[i + seq_length]
    
    # Convert to PyTorch tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.FloatTensor(X).to(device)
    y = torch.FloatTensor(y).to(device)
    
    return X, y

def split_data(X, y):
    """
    Split data into training, validation, and test sets.
    
    Args:
        X (torch.Tensor): Input sequences
        y (torch.Tensor): Target sequences
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: separate test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, 
        test_size=config.TRAINING_CONFIG['test_size'],
        shuffle=False
    )
    
    # Second split: separate validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=config.TRAINING_CONFIG['val_size'],
        shuffle=False
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_scaler(scaler, feature_names):
    """
    Save the scaler object for later use.
    
    Args:
        scaler (StandardScaler): Fitted scaler object
        feature_names (list): List of feature names
    """
    import joblib
    models_dir = Path(config.DATA_CONFIG['models_dir'])
    models_dir.mkdir(exist_ok=True)
    
    scaler_path = models_dir / 'scaler.joblib'
    joblib.dump((scaler, feature_names), scaler_path)
    print(f"Scaler saved to {scaler_path}")

def load_scaler():
    """
    Load the saved scaler object.
    
    Returns:
        tuple: (scaler, feature_names)
    """
    import joblib
    scaler_path = Path(config.DATA_CONFIG['models_dir']) / 'scaler.joblib'
    return joblib.load(scaler_path) 