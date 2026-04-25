import pickle

with open("prediction_result_None", "rb") as f:
    data = pickle.load(f)

print(data)