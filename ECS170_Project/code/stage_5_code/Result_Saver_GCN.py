

import os
import csv
import matplotlib.pyplot as plt


class Result_Saver_GCN:
    """
    Save training curves and evaluation results.

    Saved files:
        loss_curve.png
        accuracy_curve.png
        evaluation_metrics.csv
        classification_report.txt
    """

    def __init__(self, result_folder):
        self.result_folder = result_folder

        os.makedirs(self.result_folder, exist_ok=True)

    def save_loss_curve(self, loss_history, dataset_name):

        plt.figure(figsize=(8, 5))
        plt.plot(loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{dataset_name} Training Loss')
        plt.grid(True)

        save_path = os.path.join(
            self.result_folder,
            f'{dataset_name}_loss_curve.png'
        )

        plt.savefig(save_path)
        plt.close()

    def save_accuracy_curve(self, accuracy_history, dataset_name):

        plt.figure(figsize=(8, 5))
        plt.plot(accuracy_history)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'{dataset_name} Training Accuracy')
        plt.grid(True)

        save_path = os.path.join(
            self.result_folder,
            f'{dataset_name}_accuracy_curve.png'
        )

        plt.savefig(save_path)
        plt.close()

    def save_metrics(self, evaluation_results, dataset_name):

        save_path = os.path.join(
            self.result_folder,
            f'{dataset_name}_evaluation_metrics.csv'
        )

        with open(save_path, 'w', newline='') as file:
            writer = csv.writer(file)

            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Accuracy', evaluation_results['accuracy']])
            writer.writerow(['Precision', evaluation_results['precision']])
            writer.writerow(['Recall', evaluation_results['recall']])
            writer.writerow(['F1 Score', evaluation_results['f1']])

    def save_classification_report(self, evaluation_results, dataset_name):

        save_path = os.path.join(
            self.result_folder,
            f'{dataset_name}_classification_report.txt'
        )

        with open(save_path, 'w') as file:
            file.write(evaluation_results['classification_report'])

    def save_all_results(
        self,
        loss_history,
        accuracy_history,
        evaluation_results,
        dataset_name
    ):

        self.save_loss_curve(
            loss_history,
            dataset_name
        )

        self.save_accuracy_curve(
            accuracy_history,
            dataset_name
        )

        self.save_metrics(
            evaluation_results,
            dataset_name
        )

        self.save_classification_report(
            evaluation_results,
            dataset_name
        )

        print(f'All results saved for {dataset_name}')