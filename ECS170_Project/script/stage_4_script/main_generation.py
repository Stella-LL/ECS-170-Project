import os
import sys
import re
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd

# Add stage 4 code folder to Python path
# This script is inside: ECS170_Project/script/stage_4_script/
current_dir = os.path.dirname(os.path.abspath(__file__))
script_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(script_dir)
stage_4_code_dir = os.path.join(project_dir, "code", "stage_4_code")
sys.path.append(stage_4_code_dir)

from Method_RNN_Generation import Method_RNN_Generation


class TextGenerationDataset(Dataset):
    def __init__(self, jokes, word_to_index, sequence_length=5):
        self.input_sequences = []
        self.target_sequences = []
        self.word_to_index = word_to_index
        self.sequence_length = sequence_length

        # Create training samples from each joke.
        # Example joke words: [what, did, the, bartender, say]
        # input:  [what, did, the, bartender]
        # target: [did, the, bartender, say]
        for joke in jokes:
            token_ids = []
            for word in joke:
                token_ids.append(self.word_to_index.get(word, self.word_to_index["<UNK>"]))

            # Need enough words to create one input-target pair.
            if len(token_ids) > sequence_length:
                for i in range(len(token_ids) - sequence_length):
                    input_seq = token_ids[i:i + sequence_length]
                    target_seq = token_ids[i + 1:i + sequence_length + 1]

                    self.input_sequences.append(input_seq)
                    self.target_sequences.append(target_seq)

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, index):
        x = self.input_sequences[index]
        y = self.target_sequences[index]

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def clean_text(text):
    # Lowercase text.
    text = text.lower()

    # Put spaces around punctuation that can end a joke.
    text = re.sub(r"([.!?])", r" \1 ", text)

    # Remove other punctuation/symbols but keep letters, apostrophes, and sentence punctuation.
    text = re.sub(r"[^a-zA-Z'.!?]", " ", text)

    # Remove extra spaces.
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    return words


def load_jokes(joke_file_path):
    jokes = []

    with open(joke_file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()

        if line != "":
            words = clean_text(line)

            # Add an end token so the model can learn when to stop.
            words.append("<END>")
            jokes.append(words)

    return jokes


def build_vocabulary(jokes, max_vocab_size=10000, min_freq=1):
    word_counter = Counter()

    for joke in jokes:
        word_counter.update(joke)

    most_common_words = word_counter.most_common(max_vocab_size)

    word_to_index = {
        "<UNK>": 0
    }

    for word, count in most_common_words:
        if count >= min_freq and word not in word_to_index:
            word_to_index[word] = len(word_to_index)

    index_to_word = {}
    for word, index in word_to_index.items():
        index_to_word[index] = word

    return word_to_index, index_to_word


def save_loss_curve(loss_history, result_folder, model_name="RNN_Generation"):
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    epochs = range(1, len(loss_history) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(model_name + " Training Loss")
    plt.grid(True)

    loss_path = os.path.join(result_folder, model_name.lower() + "_loss_curve.png")
    plt.savefig(loss_path)
    plt.close()


# Save evaluation metrics as a CSV file
def save_evaluation_metrics(loss_history,
                            evaluation_results,
                            result_folder,
                            num_jokes,
                            vocab_size,
                            num_training_sequences,
                            model_name="RNN_Generation"):

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    metrics_df = pd.DataFrame({
        "Metric": [
            "Final Training Loss",
            "Best Training Loss",
            "Evaluation Loss",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "Number of Epochs",
            "Number of Jokes",
            "Vocabulary Size",
            "Number of Training Sequences"
        ],
        "Value": [
            loss_history[-1],
            min(loss_history),
            evaluation_results["loss"],
            evaluation_results["accuracy"],
            evaluation_results["precision"],
            evaluation_results["recall"],
            evaluation_results["f1"],
            len(loss_history),
            num_jokes,
            vocab_size,
            num_training_sequences
        ]
    })

    metrics_path = os.path.join(
        result_folder,
        model_name.lower() + "_evaluation_metrics.csv"
    )

    metrics_df.to_csv(metrics_path, index=False)


def save_generated_examples(generated_examples, result_folder):
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    output_path = os.path.join(result_folder, "rnn_generation_examples_1.txt")

    with open(output_path, "w", encoding="utf-8") as file:
        for start_words, generated_text in generated_examples:
            file.write("Starting words: " + start_words + "\n")
            file.write("Generated text: " + generated_text + "\n")
            file.write("\n")


def main():
    print("Stage 4 Text Generation with RNN")

    # Define paths.
    data_folder = os.path.join(project_dir, "data", "stage_4_data", "text_generation")
    result_folder = os.path.join(project_dir, "result", "stage_4_result")

    # The joke data file was provided as a plain file named data.
    joke_file_path = os.path.join(data_folder, "data")

    print("Loading jokes...")
    jokes = load_jokes(joke_file_path)
    print("Number of jokes:", len(jokes))

    print("Building vocabulary...")
    word_to_index, index_to_word = build_vocabulary(jokes, max_vocab_size=10000, min_freq=1)
    vocab_size = len(word_to_index)
    print("Vocabulary size:", vocab_size)

    # Hyperparameters.
    sequence_length = 5
    batch_size = 64
    embedding_dim = 128
    hidden_size = 128
    num_layers = 1
    learning_rate = 0.001
    epochs = 30

    print("Creating dataset and dataloader...")
    train_dataset = TextGenerationDataset(
        jokes=jokes,
        word_to_index=word_to_index,
        sequence_length=sequence_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    print("Number of training sequences:", len(train_dataset))

    print("Training RNN generation model...")
    method = Method_RNN_Generation(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        learning_rate=learning_rate,
        epochs=epochs
    )

    loss_history = method.train_model(train_loader)

    print("Evaluating generation model...")
    evaluation_results = method.evaluate_model(train_loader)

    print("Saving loss curve...")
    save_loss_curve(loss_history, result_folder, model_name="RNN_Generation")

    print("Saving evaluation metrics...")
    save_evaluation_metrics(
        loss_history=loss_history,
        evaluation_results=evaluation_results,
        result_folder=result_folder,
        num_jokes=len(jokes),
        vocab_size=vocab_size,
        num_training_sequences=len(train_dataset),
        model_name="RNN_Generation"
    )

    # Examples required by the project: three starting words.
    start_word_list = [
        "what did the",
        "why did the",
        "how do you",
        "what do you",
        "i told my"
    ]

    print("Generating jokes...")
    generated_examples = []

    for start_words in start_word_list:
        generated_text = method.generate_text(
            start_words=start_words,
            word_to_index=word_to_index,
            index_to_word=index_to_word,
            max_length=40
        )

        generated_examples.append((start_words, generated_text))
        print("Start:", start_words)
        print("Generated:", generated_text)
        print()

    print("Saving generated examples...")
    save_generated_examples(generated_examples, result_folder)

    model_path = os.path.join(result_folder, "rnn_generation_model.pth")
    method.save_model(model_path)

    print("Finished Stage 4 text generation.")


if __name__ == "__main__":
    main()