import os
import csv
import matplotlib.pyplot as plt


class Result_Saver_CNN:

    def __init__(self, result_folder='results'):
        self.result_folder = result_folder

        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)

    def save_loss_curve(self, losses, dataset_name):
        plt.figure()
        plt.plot(range(1, len(losses) + 1), losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{dataset_name} Training Loss')
        plt.savefig(f'{self.result_folder}/{dataset_name}_loss_curve.png')
        plt.close()

    def save_metrics(self, dataset_name, accuracy, precision, recall, f1):
        file_path = f'{self.result_folder}/{dataset_name}_metrics.csv'

        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Dataset', 'Accuracy', 'Precision', 'Recall', 'F1'])
            writer.writerow([dataset_name, accuracy, precision, recall, f1])

    def print_metrics(self, dataset_name, accuracy, precision, recall, f1):
        print(f'\n{dataset_name} Test Performance')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')