from code.stage_2_code.Dataset_Loader import Dataset_Loader
from code.stage_2_code.Method_MLP_Improved import Method_MLP2
from code.stage_2_code.Result_Saver import Result_Saver
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy

import numpy as np
import torch


# ---- Multi-Layer Perceptron script for Stage 2 ----
if __name__ == '__main__':

    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- object initialization section -------------------
    data_folder_path = "../ECS170_Project/data/stage_2_data/"

    train_data_obj = Dataset_Loader('stage 2 training data', '')
    train_data_obj.dataset_source_folder_path = data_folder_path
    train_data_obj.dataset_source_file_name = 'train.csv'

    test_data_obj = Dataset_Loader('stage 2 testing data', '')
    test_data_obj.dataset_source_folder_path = data_folder_path
    test_data_obj.dataset_source_file_name = 'test.csv'

    method_obj2 = Method_MLP2('multi-layer perceptron 1', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../ECS170_Project/result/stage_2_result/'
    result_obj.result_destination_file_name = 'MLP_prediction_result_improved'
    result_obj.fold_count = 0

    evaluate_obj = Evaluate_Accuracy('accuracy', '')

    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')

    print('loading training data...')
    train_data = train_data_obj.load()

    print('loading testing data...')
    test_data = test_data_obj.load()

    method_obj2.data = {
        'train': train_data,
        'test': test_data
    }

    print('************ Training and Testing ************')
    learned_result = method_obj2.run()

    print('************ Saving Result ************')
    result_obj.data = learned_result
    result_obj.save()

    print('************ Evaluation ************')
    evaluate_obj.data = learned_result
    score = evaluate_obj.evaluate()

    print('************ Overall Performance ************')
    print('MLP Accuracy:', score)

    print('************ Finish ************')
    # ------------------------------------------------------

    