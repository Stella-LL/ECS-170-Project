import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class RNN_Text_Classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, num_layers=1):
        super(RNN_Text_Classifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Convert word indices into word vectors.
        # padding_idx=0 tells PyTorch that index 0 is the <PAD> token.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # GRU reads the review from left-to-right.
        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Final classifier layer.
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: [batch_size, sequence_length]
        # Each row is one padded review.

        # Convert word indices to embeddings.
        embedded = self.embedding(x)

        # Run the full sequence through GRU.
        # rnn_out shape: [batch_size, sequence_length, hidden_size]
        rnn_out, hidden = self.rnn(embedded)

        # Architecture 2 from the slide:
        # Instead of only using the last hidden state, use all hidden states.
        # We average hidden states over the real words and ignore <PAD> positions.
        mask = (x != 0).unsqueeze(-1).to(rnn_out.device)
        masked_rnn_out = rnn_out * mask

        lengths = (x != 0).sum(dim=1).unsqueeze(1).to(rnn_out.device)
        lengths = torch.clamp(lengths, min=1)

        final_hidden = masked_rnn_out.sum(dim=1) / lengths

        # Predict negative/positive class scores.
        output = self.fc(final_hidden)

        return output


class Method_RNN:
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=128, num_classes=2, num_layers=1,
                 learning_rate=0.0005, epochs=15, device=None):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = RNN_Text_Classifier(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_classes=self.num_classes,
            num_layers=self.num_layers
        ).to(self.device)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.train_loss_history = []
        self.train_accuracy_history = []

    def train_model(self, train_loader):
        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0.0
            all_predictions = []
            all_labels = []

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(batch_x)
                loss = self.loss_function(outputs, batch_y)

                loss.backward()

                # Clip gradients to reduce exploding-gradient problems in RNN training.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

                self.optimizer.step()

                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

            avg_loss = total_loss / len(train_loader)
            train_accuracy = accuracy_score(all_labels, all_predictions)

            self.train_loss_history.append(avg_loss)
            self.train_accuracy_history.append(train_accuracy)

            print("Epoch [{}/{}], Loss: {:.4f}, Train Accuracy: {:.4f}".format(
                epoch + 1,
                self.epochs,
                avg_loss,
                train_accuracy
            ))

        return self.train_loss_history, self.train_accuracy_history

    def evaluate_model(self, test_loader):
        self.model.eval()

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                _, predicted = torch.max(outputs, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average="binary", zero_division=0)
        recall = recall_score(all_labels, all_predictions, average="binary", zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average="binary", zero_division=0)

        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "labels": all_labels,
            "predictions": all_predictions
        }

        print("Test Accuracy: {:.4f}".format(accuracy))
        print("Test Precision: {:.4f}".format(precision))
        print("Test Recall: {:.4f}".format(recall))
        print("Test F1: {:.4f}".format(f1))

        return results

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, load_path):
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))
        self.model.to(self.device)