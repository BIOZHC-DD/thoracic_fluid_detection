import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import argparse
from bioimpedance_gnn.dataset import BioimpedanceDataset
from bioimpedance_gnn.models import variant0
from bioimpedance_gnn.models import variant1
from bioimpedance_gnn.models import variant2
from bioimpedance_gnn.models import variant3
from bioimpedance_gnn.train import train_model
from bioimpedance_gnn.evaluate import test_model, calculate_metrics, plot_results

def get_model(variant, num_node_features, num_edge_features, hidden_dim, output_dim, device):
    """Returns the specified model variant."""
    if variant == "variant0":
        model = variant0(num_node_features, num_edge_features, hidden_dim, output_dim)
    elif variant == "variant1":
        model = variant1(num_node_features, num_edge_features, hidden_dim, output_dim)  # Replace with actual model
    elif variant == "variant2":
        model = variant2(num_node_features, num_edge_features, hidden_dim, output_dim)  # Replace with actual model
    elif variant == "variant3":
        model = variant3(num_node_features, hidden_dim, output_dim)  # Replace with actual model
    else:
        raise ValueError("Invalid model variant selected.")
    
    return model.to(device)

def main():
    parser = argparse.ArgumentParser(description="Train or test a GNN model for bioimpedance data.")
    parser.add_argument("--mode", choices=["train", "test"], required=True, help="Mode: train or test.")
    parser.add_argument("--variant", choices=["variant0", "variant1", "variant2", "variant3"], required=True, help="Model variant to use.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    pickle_path = 'C:/Users/ymane/Desktop/BIOZ_APOLLO_THESIS/bioimpedance_gnn/final_data_normalized_150.csv'
    dataset = BioimpedanceDataset(pickle_path)
    
    # Split dataset
    train_graphs, test_graphs = train_test_split(dataset.graphs, test_size=0.3, random_state=73)
    train_graphs, val_graphs = train_test_split(train_graphs, test_size=0.3, random_state=73)
    
    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=True)
    
    # Model parameters
    num_node_features = 7
    num_edge_features = 6
    hidden_dim = 64
    output_dim = 1
    
    model = get_model(args.variant, num_node_features, num_edge_features, hidden_dim, output_dim, device)

    if args.mode == "train":
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

        # Train model
        print(f"Starting training for {args.variant}...")
        model = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            num_epochs=200,
            device=device,
            patience=5
        )

    
    elif args.mode == "test":
        # Load trained model
        model_path = f'C:/Users/ymane/Desktop/BIOZ_APOLLO_THESIS/bioimpedance_gnn/{args.variant}.pth'
        print(f"Loading model from {model_path}...")
        model.load_state_dict(torch.load(model_path))
        
        # Evaluate model
        print(f"Evaluating {args.variant}...")
        criterion = nn.MSELoss()
        test_loss, predictions, actual_values = test_model(model, test_loader, criterion, device)
        print(f'Test Loss: {test_loss:.4f}')

        # Calculate and display metrics
        metrics = calculate_metrics(actual_values, predictions)
        print(f'R-squared: {metrics["r2"]:.4f}')
        print(f'MAE: {metrics["mae"]:.4f}')
        print(f'RMSE: {metrics["rmse"]:.4f}')

        # Plot results
        plot_results(actual_values, predictions)
        print("Evaluation complete!")

    # # Loss function and optimizer
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    # # Train model
    # print("Starting training...")
    # model = train_model(
    #     model,
    #     train_loader,
    #     val_loader,
    #     criterion,
    #     optimizer,
    #     num_epochs=200,
    #     device=device,
    #     patience=5
    # )
    
    # # Test model
    # print("Evaluating model...")
    # model.load_state_dict(torch.load('C:/Users/ymane/Desktop/BIOZ_APOLLO_THESIS/bioimpedance_gnn/our_model_varient0.pth'))
    # test_loss, predictions, actual_values = test_model(model, test_loader, criterion, device)
    # print(f'Test Loss: {test_loss:.4f}')
    
    # # Calculate and display metrics
    # metrics = calculate_metrics(actual_values, predictions)
    # print(f'R-squared: {metrics["r2"]:.4f}')
    # print(f'MAE: {metrics["mae"]:.4f}')
    # print(f'RMSE: {metrics["rmse"]:.4f}')
    
    # # Plot results
    # plot_results(actual_values, predictions)
    
    # print("Evaluation complete!")

if __name__ == "__main__":
    main()
