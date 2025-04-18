"""
Visualization module for plotting model results.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import config

# Set the style for all plots
plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 12

def plot_predictions(dates, actual, predicted, feature_names, metrics):
    # Make sure we have a place to save the plots
    results_dir = Path(config.DATA_CONFIG['results_dir'])
    results_dir.mkdir(exist_ok=True)
    
    # Plot each feature separately
    for i, feature in enumerate(feature_names):
        plt.figure(figsize=(8, 4))
        
        # Plot the actual values in black
        plt.plot(
            dates[-len(actual):], 
            actual[:, i], 
            label='Actual', 
            color='black',
            linewidth=1.5
        )
        
        # Plot the predicted values in red
        plt.plot(
            dates[-len(predicted):], 
            predicted[:, i], 
            label='Predicted', 
            color='red',
            linestyle='--',
            linewidth=1.5
        )
        
        # Add title with metrics
        plt.title(
            f'{feature} - Actual vs Predicted\n'
            f'MSE: {metrics[feature]["MSE"]:.4f}, '
            f'MAE: {metrics[feature]["MAE"]:.4f}, '
            f'MAPE: {metrics[feature]["MAPE"]:.2f}%'
        )
        
        # Add labels and make it look nice
        plt.xlabel('Date')
        plt.legend(frameon=True, framealpha=1.0)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(results_dir / f'{feature}_prediction.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_loss_curves(train_losses, val_losses):
    # Create a new figure
    plt.figure(figsize=(8, 4))
    
    # Plot both loss curves
    plt.plot(
        train_losses, 
        label='Training Loss',
        color='black',
        linewidth=1.5
    )
    plt.plot(
        val_losses, 
        label='Validation Loss',
        color='red',
        linewidth=1.5
    )
    
    # Add labels and title
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(frameon=True, framealpha=1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    results_dir = Path(config.DATA_CONFIG['results_dir'])
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_results_summary(actual, predicted, feature_names, metrics):
    """Create a summary plot of model error metrics"""
    # Create shortened names mapping
    name_mapping = {
        'Unemployment Rate': 'UR',
        'Labor Force Participation Rate': 'LFPR',
        'Employment-Population (Men)': 'EPR (M)',
        'Employment-Population (Women)': 'EPR (W)',
        'U-6 Unemployment Rate (Underemployment)': 'U6'
    }
    
    n_features = len(feature_names)
    plt.figure(figsize=(10, 5))
    
    x = np.arange(n_features)
    width = 0.25
    
    mape_values = [metrics[feature]['MAPE'] for feature in feature_names]
    mae_values = [metrics[feature]['MAE'] for feature in feature_names]
    mse_values = [metrics[feature]['MSE'] for feature in feature_names]
    
    # Create bars with different patterns
    plt.bar(x - width, mape_values, width, label='MAPE (%)', 
           color='#f0f0f0', edgecolor='black', hatch='')
    plt.bar(x, mae_values, width, label='MAE',
           color='black', hatch='')
    plt.bar(x + width, mse_values, width, label='MSE',
           color='white', edgecolor='black', hatch='xxx')
    
    # Add only MAPE value labels
    label_offset = 0.1
    for i, mape in enumerate(mape_values):
        plt.text(x[i] - width, mape + label_offset, f'{mape:.1f}%', 
                ha='center', va='bottom', fontsize=8)
    
    plt.title('Model Performance Metrics by Feature')
    shortened_names = [name_mapping.get(name, name) for name in feature_names]
    plt.xticks(x, shortened_names, rotation=45, ha='right')
    plt.ylabel('Error Value')
    plt.legend(frameon=True, framealpha=1.0, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add more top margin
    plt.margins(y=0.2)
    
    plt.tight_layout()
    
    # Save the plot
    results_dir = Path(config.DATA_CONFIG['results_dir'])
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / 'model_results_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_metrics(metrics):
    # Print metrics in a nice format
    print("\nFinal Metrics:")
    for feature, feature_metrics in metrics.items():
        print(f"\n{feature}:")
        print(f"  MSE: {feature_metrics['MSE']:.4f}")
        print(f"  MAE: {feature_metrics['MAE']:.4f}")
        print(f"  MAPE: {feature_metrics['MAPE']:.2f}%") 