

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# RNN model for text generation.
# Instead of predicting a class label, this model predicts the next word.
class RNN_Text_Generator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1):
        super(RNN_Text_Generator, self).__init__()

        # Convert word indices into embedding vectors.
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # RNN layer.
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Final linear layer predicts the next word.
        # Output size = vocabulary size because we predict one word from vocabulary.
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        # x shape: [batch_size, sequence_length]

        # Convert word IDs into embedding vectors.
        embedded = self.embedding(x)

        # Pass sequence through RNN.
        rnn_out, hidden = self.rnn(embedded, hidden)

        # Convert hidden states into vocabulary prediction scores.
        output = self.fc(rnn_out)

        return output, hidden


class Method_RNN_Generation:
    def __init__(self,
                 vocab_size,
                 embedding_dim=128,
                 hidden_size=128,
                 num_layers=1,
                 learning_rate=0.001,
                 epochs=20,
                 device=None):

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Use GPU if available.
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Create RNN generation model.
        self.model = RNN_Text_Generator(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        ).to(self.device)

        # CrossEntropyLoss is used for next-word prediction.
        self.loss_function = nn.CrossEntropyLoss()

        # Adam optimizer.
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Save training loss for learning curve.
        self.train_loss_history = []

    def train_model(self, train_loader):
        # Set model to training mode.
        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0.0

            # Each batch contains:
            # input sequence and target sequence.
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Clear old gradients.
                self.optimizer.zero_grad()

                # Forward pass.
                outputs, hidden = self.model(batch_x)

                # Reshape output for CrossEntropyLoss.
                # outputs shape:
                # [batch_size, sequence_length, vocab_size]
                outputs = outputs.reshape(-1, self.vocab_size)

                # Flatten targets.
                batch_y = batch_y.reshape(-1)

                # Compute loss.
                loss = self.loss_function(outputs, batch_y)

                # Backpropagation.
                loss.backward()

                # Update weights.
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            self.train_loss_history.append(avg_loss)

            print("Epoch [{}/{}], Loss: {:.4f}".format(
                epoch + 1,
                self.epochs,
                avg_loss
            ))

        return self.train_loss_history

    def evaluate_model(self, test_loader):
        # Set model to evaluation mode.
        self.model.eval()

        all_predictions = []
        all_labels = []
        total_loss = 0.0

        # No gradient calculation during evaluation.
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass.
                outputs, hidden = self.model(batch_x)

                # outputs shape: [batch_size, sequence_length, vocab_size]
                # Convert to [batch_size * sequence_length, vocab_size]
                outputs_for_loss = outputs.reshape(-1, self.vocab_size)

                # batch_y shape: [batch_size, sequence_length]
                # Convert to [batch_size * sequence_length]
                labels_for_loss = batch_y.reshape(-1)

                # Calculate loss.
                loss = self.loss_function(outputs_for_loss, labels_for_loss)
                total_loss += loss.item()

                # Pick predicted next word for every position.
                _, predicted = torch.max(outputs_for_loss, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels_for_loss.cpu().numpy())

        avg_loss = total_loss / len(test_loader)

        # Next-word prediction is multi-class classification over vocabulary words.
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average="macro", zero_division=0)
        recall = recall_score(all_labels, all_predictions, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)

        results = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

        print("Generation Evaluation Loss: {:.4f}".format(avg_loss))
        print("Generation Accuracy: {:.4f}".format(accuracy))
        print("Generation Precision: {:.4f}".format(precision))
        print("Generation Recall: {:.4f}".format(recall))
        print("Generation F1: {:.4f}".format(f1))

        return results

    def generate_text(self,
                      start_words,
                      word_to_index,
                      index_to_word,
                      max_length=50):

        # Set model to evaluation mode.
        self.model.eval()

        # Convert starting words into indices.
        current_words = start_words.lower().split()

        generated_words = current_words.copy()

        # Hidden state starts as None.
        hidden = None

        with torch.no_grad():
            for _ in range(max_length):

                # Convert words into IDs.
                input_ids = []
                for word in current_words:
                    input_ids.append(word_to_index.get(word, word_to_index["<UNK>"]))

                # Shape becomes [1, sequence_length]
                input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)

                # Get prediction from model.
                outputs, hidden = self.model(input_tensor, hidden)

                # Take prediction from the last word position.
                last_output = outputs[:, -1, :]

                # Pick the word with highest score.
                predicted_index = torch.argmax(last_output, dim=1).item()

                # Convert index back into word.
                predicted_word = index_to_word[predicted_index]

                # Stop generation if end token appears.
                if predicted_word == "<END>":
                    break

                # Add predicted word to final sentence.
                generated_words.append(predicted_word)

                # Update current words.
                current_words.append(predicted_word)

                # Keep sequence from becoming too long.
                current_words = current_words[-20:]

        return " ".join(generated_words)

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, load_path):
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))
        self.model.to(self.device)