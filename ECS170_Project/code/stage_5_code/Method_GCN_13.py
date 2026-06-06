import random
import numpy as np
import torch

from Method_GCN_2 import Method_GCN
from utils import load_data, create_balanced_split, save_results

print("Stage 5: Graph Embedding and Node Classification with GCN - Model 13 (Seed 2)")

random.seed(10)
np.random.seed(10)
torch.manual_seed(10)

# Hyperparameters
hidden_size = 32
learning_rate = 0.005
weight_decay = 5e-4
epochs = 300
dropout = 0.3

# Load dataset
dataset_name = "cora"
features, adj, labels, class_names = load_data(dataset_name)

# Create train/test split
idx_train, idx_test = create_balanced_split(labels, seed=10)

print("Model Version: Model 13 (Model 2 + Seed 2)")

# Initialize model
method = Method_GCN(
    input_size=features.shape[1],
    hidden_size=hidden_size,
    output_size=len(class_names),
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    epochs=epochs,
    dropout=dropout,
)

# Train model
train_loss_history, train_accuracy_history = method.train_model(features, adj, labels, idx_train)

# Evaluate model
evaluation_results = method.evaluate_model(features, adj, labels, idx_test, class_names)

# Save results
result_folder = "model13"
model_save_path = f"{dataset_name}_gcn_model13.pth"
save_results(result_folder, model_save_path, method, evaluation_results, train_loss_history, train_accuracy_history)