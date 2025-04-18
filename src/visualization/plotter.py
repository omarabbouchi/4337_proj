"""
Visualization module for plotting model results.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import config

# Set the style for all plots
plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.family'] = config.VISUALIZATION_CONFIG['font_family']
plt.rcParams['font.size'] = config.VISUALIZATION_CONFIG['font_size']
plt.rcParams['axes.labelsize'] = config.VISUALIZATION_CONFIG['label_size']
plt.rcParams['axes.titlesize'] = config.VISUALIZATION_CONFIG['title_size']
plt.rcParams['xtick.labelsize'] = config.VISUALIZATION_CONFIG['tick_size']
plt.rcParams['ytick.labelsize'] = config.VISUALIZATION_CONFIG['tick_size']
plt.rcParams['legend.fontsize'] = config.VISUALIZATION_CONFIG['legend_size']

def plot_predictions(dates, actual, predicted, feature_names, metrics):
    # Make sure we have a place to save the plots
    results_dir = Path(config.DATA_CONFIG['results_dir'])
    results_dir.mkdir(exist_ok=True)
    
    # Plot each feature separately
    for i, feature in enumerate(feature_names):
        plt.figure(figsize=config.VISUALIZATION_CONFIG['figure_size'])
        
        # Plot the actual values
        plt.plot(
            dates[-len(actual):], 
            actual[:, i], 
            label='Actual', 
            color=config.VISUALIZATION_CONFIG['colors']['actual'],
            linestyle=config.VISUALIZATION_CONFIG['line_styles']['actual'],
            linewidth=config.VISUALIZATION_CONFIG['line_width']
        )
        
        # Plot the predicted values
        plt.plot(
            dates[-len(predicted):], 
            predicted[:, i], 
            label='Predicted', 
            color=config.VISUALIZATION_CONFIG['colors']['predicted'],
            linestyle=config.VISUALIZATION_CONFIG['line_styles']['predicted'],
            linewidth=config.VISUALIZATION_CONFIG['line_width']
        )
        
        # Add title with metrics if enabled
        if config.VISUALIZATION_CONFIG['show_titles']:
            plt.title(
                f'{feature} - Actual vs Predicted\n'
                f'MSE: {metrics[feature]["MSE"]:.4f}, '
                f'MAE: {metrics[feature]["MAE"]:.4f}, '
                f'MAPE: {metrics[feature]["MAPE"]:.2f}%'
            )
        
        # Add labels and make it look nice
        plt.xlabel('Date')
        plt.legend(frameon=True, framealpha=1.0, loc=config.VISUALIZATION_CONFIG['legend_loc'])
        plt.xticks(rotation=config.VISUALIZATION_CONFIG['x_rotation'])
        plt.grid(True, linestyle=config.VISUALIZATION_CONFIG['grid_style'], 
                alpha=config.VISUALIZATION_CONFIG['grid_alpha'])
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(results_dir / f'{feature}_prediction.png', 
                   dpi=config.VISUALIZATION_CONFIG['dpi'], 
                   bbox_inches='tight')
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
    plt.figure(figsize=config.VISUALIZATION_CONFIG['figure_size'])
    
    x = np.arange(n_features)
    width = config.VISUALIZATION_CONFIG['bar_width']
    
    mape_values = [metrics[feature]['MAPE'] for feature in feature_names]
    mae_values = [metrics[feature]['MAE'] for feature in feature_names]
    mse_values = [metrics[feature]['MSE'] for feature in feature_names]
    
    # Create bars with different patterns
    plt.bar(x - width, mape_values, width, label='MAPE (%)', 
           color=config.VISUALIZATION_CONFIG['colors']['mape_bar'],
           edgecolor='black',
           hatch=config.VISUALIZATION_CONFIG['bar_patterns']['mape'])
    plt.bar(x, mae_values, width, label='MAE',
           color=config.VISUALIZATION_CONFIG['colors']['mae_bar'],
           hatch=config.VISUALIZATION_CONFIG['bar_patterns']['mae'])
    plt.bar(x + width, mse_values, width, label='MSE',
           color=config.VISUALIZATION_CONFIG['colors']['mse_bar'],
           edgecolor='black',
           hatch=config.VISUALIZATION_CONFIG['bar_patterns']['mse'])
    
    # Add MAPE value labels if enabled
    if config.VISUALIZATION_CONFIG['show_value_labels']:
        for i, mape in enumerate(mape_values):
            plt.text(x[i] - width, 
                    mape + config.VISUALIZATION_CONFIG['label_offset'],
                    f'{mape:.{config.VISUALIZATION_CONFIG["value_label_decimals"]["mape"]}f}%',
                    ha='center', va='bottom',
                    fontsize=config.VISUALIZATION_CONFIG['tick_size'])
    
    # Add title if enabled
    if config.VISUALIZATION_CONFIG['show_titles']:
        plt.title('Model Performance Metrics by Feature')
    
    shortened_names = [name_mapping.get(name, name) for name in feature_names]
    plt.xticks(x, shortened_names, 
               rotation=config.VISUALIZATION_CONFIG['x_rotation'],
               ha='right')
    plt.ylabel('Error Value')
    plt.legend(frameon=True, framealpha=1.0,
              loc=config.VISUALIZATION_CONFIG['legend_loc'])
    plt.grid(True, linestyle=config.VISUALIZATION_CONFIG['grid_style'],
            alpha=config.VISUALIZATION_CONFIG['grid_alpha'],
            axis='y')
    
    plt.margins(y=config.VISUALIZATION_CONFIG['margin_top'])
    plt.tight_layout()
    
    # Save the plot
    results_dir = Path(config.DATA_CONFIG['results_dir'])
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / 'model_results_summary.png',
                dpi=config.VISUALIZATION_CONFIG['dpi'],
                bbox_inches='tight')
    plt.close()

def print_metrics(metrics):
    # Print metrics in a nice format
    print("\nFinal Metrics:")
    for feature, feature_metrics in metrics.items():
        print(f"\n{feature}:")
        print(f"  MSE: {feature_metrics['MSE']:.4f}")
        print(f"  MAE: {feature_metrics['MAE']:.4f}")
        print(f"  MAPE: {feature_metrics['MAPE']:.2f}%") 