from code.stage_2_code.Dataset_Loader import Dataset_Loader
from code.stage_2_code.Method_MLP import Method_MLP
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# 1. Load training data
train_dataset = Dataset_Loader('training dataset', '')
train_dataset.dataset_source_folder_path = './'
train_dataset.dataset_source_file_name = '../../data/stage_2_data/train.csv'
train_data = train_dataset.load()

# 2. Load testing data
test_dataset = Dataset_Loader('testing dataset', '')
test_dataset.dataset_source_folder_path = './'
test_dataset.dataset_source_file_name = '../../data/stage_2_data/test.csv'
test_data = test_dataset.load()

# 3. Initialize MLP model
method = Method_MLP('mlp', '')

# 4. Put data into method
method.data = {
    'train': {
        'X': train_data['X'],
        'y': train_data['y']
    },
    'test': {
        'X': test_data['X'],
        'y': test_data['y']
    }
}

# 5. Train and test model
learned_result = method.run()

pred_y = learned_result['pred_y']
true_y = learned_result['true_y']

# 6. Evaluation metrics
accuracy = accuracy_score(true_y, pred_y)
precision = precision_score(true_y, pred_y, average='macro')
recall = recall_score(true_y, pred_y, average='macro')
f1 = f1_score(true_y, pred_y, average='macro')

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)