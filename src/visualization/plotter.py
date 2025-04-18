"""
Visualization module for plotting model results.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import config

def plot_predictions(dates, actual, predicted, feature_names, metrics):
    # Make sure we have a place to save the plots
    results_dir = Path(config.DATA_CONFIG['results_dir'])
    results_dir.mkdir(exist_ok=True)
    
    # Plot each feature separately
    for i, feature in enumerate(feature_names):
        plt.figure(figsize=config.VISUALIZATION_CONFIG['figure_size'])
        
        # Plot the actual values in blue
        plt.plot(
            dates[-len(actual):], 
            actual[:, i], 
            label='Actual', 
            color=config.VISUALIZATION_CONFIG['colors']['actual'],
            linewidth=config.VISUALIZATION_CONFIG['line_width']
        )
        
        # Plot the predicted values in red
        plt.plot(
            dates[-len(predicted):], 
            predicted[:, i], 
            label='Predicted', 
            color=config.VISUALIZATION_CONFIG['colors']['predicted'],
            linestyle='--',
            linewidth=config.VISUALIZATION_CONFIG['line_width']
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
        plt.ylabel('Value')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(results_dir / f'{feature}_prediction.png')
        plt.close()

def plot_loss_curves(train_losses, val_losses):
    # Create a new figure
    plt.figure(figsize=config.VISUALIZATION_CONFIG['figure_size'])
    
    # Plot both loss curves
    plt.plot(
        train_losses, 
        label='Training Loss',
        linewidth=config.VISUALIZATION_CONFIG['line_width']
    )
    plt.plot(
        val_losses, 
        label='Validation Loss',
        linewidth=config.VISUALIZATION_CONFIG['line_width']
    )
    
    # Add labels and title
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Save the plot
    results_dir = Path(config.DATA_CONFIG['results_dir'])
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / 'loss_curves.png')
    plt.close()

def print_metrics(metrics):
    # Print metrics in a nice format
    print("\nFinal Metrics:")
    for feature, feature_metrics in metrics.items():
        print(f"\n{feature}:")
        print(f"  MSE: {feature_metrics['MSE']:.4f}")
        print(f"  MAE: {feature_metrics['MAE']:.4f}")
        print(f"  MAPE: {feature_metrics['MAPE']:.2f}%") 