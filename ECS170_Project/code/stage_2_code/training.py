from code.stage_2_code.Dataset_Loader import Dataset_Loader
from code.stage_2_code.Method_MLP import Method_MLP
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy


# Load  data

dataset_folder_path = "./"
train_file_path = "../../data/stage_2_data/train.csv"
test_file_path = "../../data/stage_2_data/test.csv"

train_dataset = Dataset_Loader('train dataset','')
train_dataset.dataset_source_folder_path = dataset_folder_path
train_dataset.dataset_source_file_name = train_file_path

test_dataset = Dataset_Loader('test dataset', '')
test_dataset.dataset_source_folder_path = dataset_folder_path
test_dataset.dataset_source_file_name = test_file_path

train_data = train_dataset.load()
test_data = test_dataset.load()


# # 3. Create method
# method_obj = Method_MLP(
#     mName='MLP',
#     mDescription='Multilayer Perceptron for multiclass classification'
# )

# method_obj.data = {
#     'train': train_data,
#     'test': test_data
# }


# # 4. Train and test
# result = method_obj.run()


# # 5. Evaluate result
# evaluate_obj = Evaluate_Accuracy(
#     eName='accuracy evaluator',
#     eDescription='evaluate prediction accuracy'
# )

# evaluate_obj.data = {
#     'true_y': result['true_y'],
#     'pred_y': result['pred_y']
# }

# accuracy = evaluate_obj.evaluate()

# print('Final Testing Accuracy:', accuracy)