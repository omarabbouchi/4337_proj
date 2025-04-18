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
    # Figure settings
    'figure_size': (10, 5),
    'dpi': 300,
    
    # Font settings
    'font_family': 'Times New Roman',
    'font_size': 10,
    'title_size': 12,
    'label_size': 10,
    'tick_size': 8,
    'legend_size': 8,
    
    # Colors and styles
    'colors': {
        'actual': 'black',
        'predicted': 'red',
        'mape_bar': '#f0f0f0',
        'mae_bar': 'black',
        'mse_bar': 'white'
    },
    'line_styles': {
        'actual': '-',
        'predicted': '--'
    },
    'line_width': 1.5,
    'grid_alpha': 0.7,
    'grid_style': '--',
    
    # Bar chart settings
    'bar_width': 0.25,
    'bar_patterns': {
        'mape': '',
        'mae': '',
        'mse': 'xxx'
    },
    
    # Label settings
    'show_titles': True,  # Global control for showing/hiding titles
    'show_value_labels': True,  # Control for showing/hiding value labels
    'value_label_decimals': {
        'mape': 1,  # For percentage values
        'mae': 3,
        'mse': 3
    },
    'label_offset': 0.1,  # Vertical offset for bar value labels
    
    # Layout settings
    'margin_top': 0.2,
    'x_rotation': 45,
    'legend_loc': 'upper right'
} 