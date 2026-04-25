from code.stage_2_code.Dataset_Loader import Dataset_Loader
from code.stage_2_code.Method_MLP import Method_MLP
from code.stage_2_code.Result_Saver import Result_Saver
#from code.stage_2_code.Setting_KFold_CV import Setting_KFold_CV
#from code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------
    dataset_folder_path = "../../data/stage_2_data/"
    train_file_name = "train.csv"
    test_file_name = "test.csv"
    # ---- objection initialization setction ---------------

    # ---- load training data ----
    train_data_obj = Dataset_Loader("train data", "")
    train_data_obj.dataset_source_folder_path = dataset_folder_path
    train_data_obj.dataset_source_file_name = train_file_name
    train_data = train_data_obj.load()

    # ---- load testing data ----
    test_data_obj = Dataset_Loader("test data", "")
    test_data_obj.dataset_source_folder_path = dataset_folder_path
    test_data_obj.dataset_source_file_name = test_file_name
    test_data = test_data_obj.load()

    method_obj = Method_MLP('multi-layer perceptron', '')
    method_obj.data = {
        "train": {
            "X": train_data["X"],
            "y": train_data["y"]
        },
        "test": {
            "X": test_data["X"],
            "y": test_data["y"]
        }
    }
    # ---- train and test ----
    print("************ Start ************")
    learned_result = method_obj.run()
    loss_list = learned_result["loss_list"]
    # ---- save prediction results ----
    result_obj = Result_Saver("saver", "")
    result_obj.result_destination_folder_path = "./"
    result_obj.result_destination_file_name = "prediction_result"
    result_obj.data = learned_result
    result_obj.save()

    #setting_obj = Setting_KFold_CV('k fold cross validation', '')
    #setting_obj = Setting_Tra
    # in_Test_Split('train test split', '')

    # ---- evaluation ----
    pred_y = learned_result["pred_y"]
    true_y = learned_result["true_y"]

    # Evaluate_Accuracy
    evaluate_obj = Evaluate_Accuracy("accuracy", "")
    evaluate_obj.data = {
        "true_y": true_y,
        "pred_y": pred_y
    }
    accuracy = evaluate_obj.evaluate()

    #  use sklearn to get metrics
    precision = precision_score(true_y, pred_y, average="macro")
    recall = recall_score(true_y, pred_y, average="macro")
    f1 = f1_score(true_y, pred_y, average="macro")

    print("************ Overall Performance ************")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    # ------------------------------------------------------
    # ---- training convergence plot ----
    plt.figure(figsize=(8, 6))
    plt.plot(loss_list)
    plt.xlabel("Training Epoch")
    plt.ylabel("Loss")
    plt.title("Training Convergence Plot")
    plt.savefig("training_convergence_plot_model_1.png")
    plt.show()
    # ---- confusion matrix ----
    cm = confusion_matrix(true_y, pred_y)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()
    plt.savefig("confusion_matrix_model_1.png")
    plt.show()

    print("************ Finish ************")

    

    