import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def test_model(model, test_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    predictions = []
    actual_values = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(output, data.y)
            total_loss += loss.item()

            predictions.extend(output.cpu().numpy().flatten())  # Store predictions
            actual_values.extend(data.y.cpu().numpy().flatten())  # Store actual values

    avg_loss = total_loss / len(test_loader)
    return avg_loss, predictions, actual_values

def calculate_metrics(actual_values, predictions):
    """Calculate and return evaluation metrics."""
    r2 = r2_score(actual_values, predictions)
    mae = mean_absolute_error(actual_values, predictions)
    rmse = np.sqrt(mean_squared_error(actual_values, predictions))
    
    return {
        'r2': r2,
        'mae': mae,
        'rmse': rmse
    }

def plot_results(actual_values, predictions, title="Actual vs Predicted Values"):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(actual_values, predictions, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(min(actual_values), min(predictions))
    max_val = max(max(actual_values), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.grid(True)
    
    # Add metrics to plot
    metrics = calculate_metrics(actual_values, predictions)
    plt.text(
        0.05, 0.95, 
        f"R² = {metrics['r2']:.4f}\nMAE = {metrics['mae']:.4f}\nRMSE = {metrics['rmse']:.4f}", 
        transform=plt.gca().transAxes,
        bbox=dict(facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    plt.savefig('prediction_results.png')
    plt.show()