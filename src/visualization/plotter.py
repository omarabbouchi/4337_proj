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

def plot_error_summary(metrics):
    """Create a summary bar plot of error metrics across all features"""
    features = list(metrics.keys())
    mape_values = [metrics[feature]['MAPE'] for feature in features]
    
    plt.figure(figsize=(10, 4))
    bars = plt.bar(features, mape_values, color='gray')
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.title('Mean Absolute Percentage Error (MAPE) by Feature')
    plt.ylabel('MAPE (%)')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    results_dir = Path(config.DATA_CONFIG['results_dir'])
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / 'error_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_results_summary(actual, predicted, feature_names, metrics):
    """Create a comprehensive summary plot of model results"""
    n_features = len(feature_names)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Actual vs Predicted for all features
    for i, feature in enumerate(feature_names):
        axes[0].plot(
            actual[:, i], 
            label=f'Actual {feature}',
            color='black',
            linewidth=1.5
        )
        axes[0].plot(
            predicted[:, i], 
            label=f'Predicted {feature}',
            color='red',
            linestyle='--',
            linewidth=1.5
        )
    
    axes[0].set_title('Actual vs Predicted Values')
    axes[0].legend(frameon=True, framealpha=1.0, ncol=2)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Error metrics comparison
    x = np.arange(n_features)
    width = 0.25
    
    mape_values = [metrics[feature]['MAPE'] for feature in feature_names]
    mae_values = [metrics[feature]['MAE'] for feature in feature_names]
    mse_values = [metrics[feature]['MSE'] for feature in feature_names]
    
    axes[1].bar(x - width, mape_values, width, label='MAPE (%)', color='gray')
    axes[1].bar(x, mae_values, width, label='MAE', color='lightgray')
    axes[1].bar(x + width, mse_values, width, label='MSE', color='darkgray')
    
    axes[1].set_title('Error Metrics Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(feature_names, rotation=45)
    axes[1].legend(frameon=True, framealpha=1.0)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
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