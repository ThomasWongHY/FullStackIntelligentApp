import joblib
import numpy as np

model = joblib.load("nn_model.joblib")

input_arr = [1, 1, 1, 2, 1, 2, 1, 3]
num_arr = [4.0, 2.0]

pred_arr = np.array(num_arr + input_arr).reshape(1,-1)
prediction = (model.predict(pred_arr) > 0.5).astype("int32")[0][0]
print(prediction)