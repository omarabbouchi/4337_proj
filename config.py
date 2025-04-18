"""
Configuration settings for the LSTNet model and training process
"""

# Model parameters
MODEL_CONFIG = {
    'num_features': 5,  # Default number of features
    'conv_out_channels': 32,
    'gru_hidden_size': 64,
    'skip_lengths': [4, 24],  # Weekly and monthly patterns
    'skip_hidden_size': 16,
    'ar_window': 7,  # Autoregressive window size
    'dropout': 0.2
}

# Training parameters
TRAINING_CONFIG = {
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'sequence_length': 12,  # Using 12 months as sequence length
    'test_size': 0.2,
    'val_size': 0.2
}

# Data parameters
DATA_CONFIG = {
    'default_data_file': 'multivariate_unemployment_LSTNet.csv',
    'data_dir': 'data',
    'models_dir': 'models',
    'results_dir': 'results'
}

# Visualization parameters
VISUALIZATION_CONFIG = {
    'figure_size': (12, 6),
    'font_size': 12,
    'line_width': 2,
    'colors': {
        'actual': 'blue',
        'predicted': 'red'
    }
} 