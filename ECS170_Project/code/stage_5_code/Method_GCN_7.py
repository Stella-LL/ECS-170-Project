import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


class GCNLayer(nn.Module):
    """
    One Graph Convolution Layer.

    Main idea:
        1. First transform node features with a weight matrix: XW
        2. Then aggregate neighbor information using adjacency matrix: A(XW)

    Formula:
        H = A X W
    """

    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()

        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x shape: number_of_nodes x number_of_features
        # adj shape: number_of_nodes x number_of_nodes

        # Step 1: apply linear transformation to node features
        support = self.linear(x)

        # Step 2: aggregate information from neighbors
        # If adj is sparse tensor, use torch.spmm
        if adj.is_sparse:
            output = torch.spmm(adj, support)
        else:
            output = torch.mm(adj, support)

        return output


class GCN_Node_Classifier(nn.Module):
    """
    Two-layer GCN model for node classification.

    Architecture:
        input node features
            -> GCN layer
            -> ReLU
            -> Dropout
            -> GCN layer
            -> class scores
    """

    def __init__(self, input_size, hidden_size, output_size, dropout=0.5):
        super(GCN_Node_Classifier, self).__init__()

        self.gcn_layer_1 = GCNLayer(input_size, hidden_size)
        self.gcn_layer_2 = GCNLayer(hidden_size, output_size)

        self.dropout = dropout

    def forward(self, x, adj):
        # First GCN layer
        x = self.gcn_layer_1(x, adj)
        x = F.relu(x)

        # Dropout helps reduce overfitting
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second GCN layer outputs class scores for each node
        x = self.gcn_layer_2(x, adj)

        return x


class Method_GCN:
    """
    Method class for training and evaluating GCN node classification model.

    This class is designed to be similar to Stage 4 Method_GRU_Classification:
        - create model
        - train model
        - record loss and accuracy history
        - evaluate model on test nodes
    """

    def __init__(self, input_size, hidden_size=32, output_size=None,
                 learning_rate=0.005, weight_decay=5e-4,
                 epochs=300, dropout=0.2, device=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dropout = dropout

        print("=" * 60)
        print("GCN Model 7 Configuration")
        print(f"Hidden Size: {hidden_size}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Epochs: {epochs}")
        print(f"Dropout: {dropout}")
        print("=" * 60)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = GCN_Node_Classifier(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            dropout=self.dropout
        ).to(self.device)

        self.loss_function = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        self.train_loss_history = []
        self.train_accuracy_history = []

    def train_model(self, features, adj, labels, idx_train):
        """
        Train GCN model.

        Important difference from RNN:
            GCN predicts labels for all nodes at once.
            But during training, we only calculate loss on training nodes.
        """

        features = features.to(self.device)
        adj = adj.to(self.device)
        labels = labels.to(self.device)
        idx_train = idx_train.to(self.device)

        for epoch in range(self.epochs):
            self.model.train()

            self.optimizer.zero_grad()

            # Forward pass: output shape = number_of_nodes x number_of_classes
            outputs = self.model(features, adj)

            # Only use training nodes to compute loss
            loss = self.loss_function(outputs[idx_train], labels[idx_train])

            loss.backward()
            self.optimizer.step()

            # Calculate training accuracy
            _, predicted = torch.max(outputs[idx_train], dim=1)
            train_accuracy = accuracy_score(
                labels[idx_train].detach().cpu().numpy(),
                predicted.detach().cpu().numpy()
            )

            self.train_loss_history.append(loss.item())
            self.train_accuracy_history.append(train_accuracy)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{self.epochs}], "
                    f"Loss: {loss.item():.4f}, "
                    f"Train Accuracy: {train_accuracy:.4f}"
                )

        return self.train_loss_history, self.train_accuracy_history

    def evaluate_model(self, features, adj, labels, idx_test, class_names=None):
        """
        Evaluate GCN model on testing nodes.
        """

        features = features.to(self.device)
        adj = adj.to(self.device)
        labels = labels.to(self.device)
        idx_test = idx_test.to(self.device)

        self.model.eval()

        with torch.no_grad():
            outputs = self.model(features, adj)
            test_outputs = outputs[idx_test]
            test_labels = labels[idx_test]

            _, predicted = torch.max(test_outputs, dim=1)

        y_true = test_labels.detach().cpu().numpy()
        y_pred = predicted.detach().cpu().numpy()

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        report = classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0
        )

        cm = confusion_matrix(y_true, y_pred)

        evaluation_results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "classification_report": report,
            "confusion_matrix": cm,
            "y_true": y_true,
            "y_pred": y_pred
        }

        print("Testing Results:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print("\nClassification Report:")
        print(report)

        return evaluation_results

    def get_model(self):
        return self.model

    def get_train_loss_history(self):
        return self.train_loss_history

    def get_train_accuracy_history(self):
        return self.train_accuracy_history