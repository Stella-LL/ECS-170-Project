

import os
import sys
import random

import numpy as np
import torch


# Make results reproducible
random.seed(3)
np.random.seed(3)
torch.manual_seed(3)


# This script is inside: ECS170_Project/script/stage_5_script/
current_dir = os.path.dirname(os.path.abspath(__file__))
script_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(script_dir)

# Add stage 5 code folder to Python path
stage_5_code_dir = os.path.join(project_dir, "code", "stage_5_code")
sys.path.append(stage_5_code_dir)

# Add project root to Python path so Dataset_Loader can import base_class
sys.path.append(project_dir)

from Dataset_Loader_Node_Classification import Dataset_Loader
from Method_GCN_12 import Method_GCN
from Result_Saver_GCN import Result_Saver_GCN


def create_balanced_split(labels, train_per_class, test_per_class, seed=42):
    """
    Create class-balanced training and testing indices.

    Requirement from Stage 5 README:
        Cora:     20 train/class, 150 test/class
        Citeseer: 20 train/class, 200 test/class
        Pubmed:   20 train/class, 200 test/class

    labels: torch.LongTensor, shape = number_of_nodes
    """

    rng = np.random.default_rng(seed)

    labels_np = labels.cpu().numpy()
    unique_classes = np.unique(labels_np)

    train_indices = []
    test_indices = []

    for class_label in unique_classes:
        class_indices = np.where(labels_np == class_label)[0]
        rng.shuffle(class_indices)

        if len(class_indices) < train_per_class + test_per_class:
            raise ValueError(
                f"Class {class_label} does not have enough nodes. "
                f"Need {train_per_class + test_per_class}, "
                f"but only found {len(class_indices)}."
            )

        class_train = class_indices[:train_per_class]
        class_test = class_indices[train_per_class:train_per_class + test_per_class]

        train_indices.extend(class_train)
        test_indices.extend(class_test)

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    idx_train = torch.LongTensor(train_indices)
    idx_test = torch.LongTensor(test_indices)

    return idx_train, idx_test


def get_split_setting(dataset_name):
    """
    Return the required train/test number per class.
    """

    if dataset_name == "cora":
        train_per_class = 20
        test_per_class = 150
    elif dataset_name == "citeseer":
        train_per_class = 20
        test_per_class = 200
    elif dataset_name == "pubmed":
        train_per_class = 20
        test_per_class = 200
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    return train_per_class, test_per_class


def get_class_names(dataset_name):
    """
    Class names for classification report.
    The order should match the encoded label order from the dataset loader.
    If the order causes an issue, you can set class_names=None in evaluate_model.
    """

    if dataset_name == "cora":
        return [
            "Case_Based",
            "Genetic_Algorithms",
            "Neural_Networks",
            "Probabilistic_Methods",
            "Reinforcement_Learning",
            "Rule_Learning",
            "Theory"
        ]
    elif dataset_name == "citeseer":
        return [
            "AI",
            "Agents",
            "DB",
            "HCI",
            "IR",
            "ML"
        ]
    elif dataset_name == "pubmed":
        return ["0", "1", "2"]
    else:
        return None


def run_one_dataset(dataset_name, data_folder, result_folder):
    """
    Load one graph dataset, train GCN, evaluate model, and save results.
    """

    print("=" * 80)
    print(f"Running GCN node classification on {dataset_name}")
    print("=" * 80)

    # Load dataset
    dataset_loader = Dataset_Loader()
    dataset_loader.dataset_name = dataset_name
    dataset_loader.dataset_source_folder_path = data_folder

    data = dataset_loader.load()

    graph = data["graph"]
    features = graph["X"]
    labels = graph["y"]
    adj = graph["utility"]["A"]

    print("Feature shape:", features.shape)
    print("Label shape:", labels.shape)
    print("Adjacency shape:", adj.shape)

    # Create balanced train/test split based on README requirement
    train_per_class, test_per_class = get_split_setting(dataset_name)

    idx_train, idx_test = create_balanced_split(
        labels=labels,
        train_per_class=train_per_class,
        test_per_class=test_per_class,
        seed=3
    )

    print("Training nodes:", len(idx_train))
    print("Testing nodes:", len(idx_test))

    # Model 14 Hyperparameters (AdamW + Seed 3)
    input_size = features.shape[1]
    hidden_size = 32
    output_size = len(torch.unique(labels))
    learning_rate = 0.005
    weight_decay = 1e-3
    epochs = 300
    dropout = 0.3

    print("Input size:", input_size)
    print("Model Version: Model 14 (AdamW + Seed 3)")
    print("Learning Rate:", learning_rate)
    print("Epochs:", epochs)
    print("Dropout:", dropout)
    print("Hidden size:", hidden_size)
    print("Output size:", output_size)

    # Create and train GCN method
    method = Method_GCN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        dropout=dropout
    )

    print("Training GCN model...")
    loss_history, accuracy_history = method.train_model(
        features=features,
        adj=adj,
        labels=labels,
        idx_train=idx_train
    )

    print("Evaluating GCN model...")
    class_names = get_class_names(dataset_name)

    evaluation_results = method.evaluate_model(
        features=features,
        adj=adj,
        labels=labels,
        idx_test=idx_test,
        class_names=class_names
    )

    # Save results for this dataset
    dataset_result_folder = os.path.join(
        result_folder,
        "model14",
        dataset_name
    )
    result_saver = Result_Saver_GCN(dataset_result_folder)

    result_saver.save_all_results(
        loss_history=loss_history,
        accuracy_history=accuracy_history,
        evaluation_results=evaluation_results,
        dataset_name=dataset_name
    )

    # Save model
    model_path = os.path.join(
        dataset_result_folder,
        f"{dataset_name}_gcn_model14.pth"
    )
    torch.save(method.get_model().state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    return evaluation_results


def main():
    print("Stage 5: Graph Embedding and Node Classification with GCN - Model 14 (AdamW + Seed 3)")

    # Folder paths
    # Expected dataset structure:
    # data/stage_5_data/cora/node
    # data/stage_5_data/cora/link
    # data/stage_5_data/citeseer/node
    # data/stage_5_data/citeseer/link
    # data/stage_5_data/pubmed/node
    # data/stage_5_data/pubmed/link
    data_root_folder = os.path.join(project_dir, "data", "stage_5_data")
    result_folder = os.path.join(project_dir, "result", "stage_5_result")

    os.makedirs(result_folder, exist_ok=True)

    datasets = [
        "cora",
        "citeseer",
        "pubmed"
    ]

    all_results = {}

    for dataset_name in datasets:
        data_folder = os.path.join(data_root_folder, dataset_name)

        if not os.path.exists(data_folder):
            print(f"Dataset folder not found: {data_folder}")
            print(f"Skipping {dataset_name}")
            continue

        evaluation_results = run_one_dataset(
            dataset_name=dataset_name,
            data_folder=data_folder,
            result_folder=result_folder
        )

        all_results[dataset_name] = evaluation_results

    print("=" * 80)
    print("Summary of Stage 5 Results")
    print("=" * 80)

    for dataset_name, results in all_results.items():
        print(f"{dataset_name}:")
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1 Score:  {results['f1']:.4f}")

    print("Finished Stage 5 GCN node classification.")


if __name__ == "__main__":
    main()