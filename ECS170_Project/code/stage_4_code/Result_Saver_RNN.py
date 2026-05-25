import os
import pandas as pd
import matplotlib.pyplot as plt


class Result_Saver_RNN:
    def __init__(self, result_folder):
        self.result_folder = result_folder

        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)

    def save_loss_curve(self, loss_history, accuracy_history, model_name="RNN"):
        epochs = range(1, len(loss_history) + 1)

        # Loss curve
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(model_name + " Training Loss")
        plt.grid(True)

        loss_path = os.path.join(
            self.result_folder,
            model_name.lower() + "_loss_curve.png"
        )

        plt.savefig(loss_path)
        plt.close()

        # Accuracy curve
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, accuracy_history)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(model_name + " Training Accuracy")
        plt.grid(True)

        accuracy_path = os.path.join(
            self.result_folder,
            model_name.lower() + "_accuracy_curve.png"
        )

        plt.savefig(accuracy_path)
        plt.close()

    def save_evaluation_results(self, evaluation_results, model_name="RNN"):
        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1"],
            "Value": [
                evaluation_results["accuracy"],
                evaluation_results["precision"],
                evaluation_results["recall"],
                evaluation_results["f1"]
            ]
        })

        metrics_path = os.path.join(
            self.result_folder,
            model_name.lower() + "_evaluation_metrics.csv"
        )

        metrics_df.to_csv(metrics_path, index=False)

    def save_all_results(self,
                         loss_history,
                         accuracy_history,
                         evaluation_results,
                         model_name="RNN"):

        self.save_loss_curve(
            loss_history,
            accuracy_history,
            model_name
        )

        self.save_evaluation_results(
            evaluation_results,
            model_name
        )

        print(model_name + " results saved successfully.")