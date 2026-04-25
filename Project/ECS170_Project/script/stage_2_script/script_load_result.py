from code.stage_2_code.Result_Loader import Result_Loader
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


if 1:
    result_obj = Result_Loader('saver', '')
    result_obj.result_destination_folder_path = '../ECS170_Project/result/stage_2_result/'
    result_obj.result_destination_file_name = 'MLP_prediction_result_improved'

    fold_count = 0

    result_obj.fold_count = fold_count
    result_obj.load()
    # print('Fold:', fold_count, ', Result:', result_obj.data)
    result = result_obj.data
    true_y = result["true_y"]
    pred_y = result["pred_y"]
    accuracy = accuracy_score(true_y, pred_y)
    precision = precision_score(true_y, pred_y, average="macro")
    recall = recall_score(true_y, pred_y, average="macro")
    f1 = f1_score(true_y, pred_y, average="macro")

    print("************ Overall Performance ************")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    loss_list = result["loss_list"]
    plt.figure()
    plt.plot( loss_list)
    plt.xlabel("Training Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Convergence Plot for Improved MLP")
    plt.grid(True)
    plt.savefig("../ECS170_Project/result/stage_2_result/training_convergence_improved.png")
    plt.show()

    cm = confusion_matrix(true_y, pred_y)
    plt.imshow(cm)
    plt.figure(figsize=(8, 8))
    plt.title("Confusion Matrix for Improved MLP")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()
    plt.savefig("../ECS170_Project/result/stage_2_result/confusion_matrix_improved.png")
    plt.show(cm)

    print("************ Finish ************")