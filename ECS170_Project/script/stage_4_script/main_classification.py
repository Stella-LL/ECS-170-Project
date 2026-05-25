

import os
import sys
import re
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader

# Add stage 4 code folder to Python path
# This script is inside: ECS170_Project/script/stage_4_script/
current_dir = os.path.dirname(os.path.abspath(__file__))
script_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(script_dir)
stage_4_code_dir = os.path.join(project_dir, "code", "stage_4_code")
sys.path.append(stage_4_code_dir)

from Method_RNN_Classification import Method_RNN
from Result_Saver_RNN import Result_Saver_RNN


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, word_to_index, max_length=300):
        self.texts = texts
        self.labels = labels
        self.word_to_index = word_to_index
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        token_ids = []
        for word in text:
            token_ids.append(self.word_to_index.get(word, self.word_to_index["<UNK>"]))

        # Cut long reviews
        token_ids = token_ids[:self.max_length]

        # Pad short reviews with 0
        while len(token_ids) < self.max_length:
            token_ids.append(self.word_to_index["<PAD>"])

        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)       # remove HTML tags
    text = re.sub(r"[^a-zA-Z']", " ", text)  # keep letters and apostrophes
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    return words


def load_imdb_folder(data_folder):
    texts = []
    labels = []

    label_map = {
        "neg": 0,
        "pos": 1
    }

    for label_name in ["neg", "pos"]:
        folder_path = os.path.join(data_folder, label_name)

        for file_name in os.listdir(folder_path):
            if file_name.endswith(".txt"):
                file_path = os.path.join(folder_path, file_name)

                with open(file_path, "r", encoding="utf-8") as file:
                    review = file.read()

                words = clean_text(review)
                texts.append(words)
                labels.append(label_map[label_name])

    return texts, labels


def build_vocabulary(texts, max_vocab_size=20000, min_freq=2):
    word_counter = Counter()

    for text in texts:
        word_counter.update(text)

    most_common_words = word_counter.most_common(max_vocab_size)

    word_to_index = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for word, count in most_common_words:
        if count >= min_freq:
            word_to_index[word] = len(word_to_index)

    return word_to_index


def main():
    print("Stage 4 Text Classification with RNN")

    # Define paths
    train_folder = os.path.join(project_dir, "data", "stage_4_data", "text_classification", "train")
    test_folder = os.path.join(project_dir, "data", "stage_4_data", "text_classification", "test")
    result_folder = os.path.join(project_dir, "result", "stage_4_result")

    print("Loading training data...")
    train_texts, train_labels = load_imdb_folder(train_folder)

    print("Loading testing data...")
    test_texts, test_labels = load_imdb_folder(test_folder)

    print("Building vocabulary...")
    word_to_index = build_vocabulary(train_texts, max_vocab_size=20000, min_freq=2)
    vocab_size = len(word_to_index)
    print("Vocabulary size:", vocab_size)

    # Hyperparameters
    max_length = 300
    batch_size = 64
    embedding_dim = 128
    hidden_size = 128
    num_classes = 2
    num_layers = 1
    learning_rate = 0.001
    epochs = 10

    print("Creating datasets and dataloaders...")
    train_dataset = TextClassificationDataset(
        train_texts,
        train_labels,
        word_to_index,
        max_length=max_length
    )

    test_dataset = TextClassificationDataset(
        test_texts,
        test_labels,
        word_to_index,
        max_length=max_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    print("Training RNN model...")
    method = Method_RNN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_classes=num_classes,
        num_layers=num_layers,
        learning_rate=learning_rate,
        epochs=epochs
    )

    loss_history, accuracy_history = method.train_model(train_loader)

    print("Evaluating RNN model...")
    evaluation_results = method.evaluate_model(test_loader)

    print("Saving results...")
    result_saver = Result_Saver_RNN(result_folder)

    result_saver.save_all_results(
        loss_history=loss_history,
        accuracy_history=accuracy_history,
        evaluation_results=evaluation_results,
        model_name="RNN_Classification"
    )

    model_path = os.path.join(result_folder, "rnn_classification_model.pth")
    method.save_model(model_path)

    print("Finished Stage 4 text classification.")


if __name__ == "__main__":
    main()