import joblib
import numpy as np

model = joblib.load("nn_model.joblib")

options_first_language = ['English', 'French', 'Other']
options_funding = ['Apprentice_PS ', 'GPOG_FT', 'Intl Offshore', 'Unknown', 
           'Intl Regular', 'Intl Transfer', 'Joint Program Ryerson', 
           'Joint Program UTSC', 'Second Career Program', 'Work Safety Insurance Board']
options_school = ['Advancement', 'Business', 'Communications', 'Community and Health', 'Hospitality', 'Engineering', 'Transportation']
options_fast_track = ['Yes', 'No']
options_coop = ['Yes', 'No']
options_residency = ['Domestic', 'International']
options_gender = ['Female', 'Male', 'Netural']
options_prev_edu = ['High School', 'Post Secondary']
options_age = ['0 to 18', '19 to 20', '21 to 25', '26 to 30', '31 to 35', '36 to 40', '41 to 50', '51 to 60', '61 to 65', '66+']
options_eng_grade = ['Level-130', 'Level-131', 'Level-140', 'Level-141', 'Level-150', 'Level-151', 'Level-160', 'Level-161', 'Level-170', 'Level-171', 'Level-180']

first_language = 'English'
funding = 'Intl Transfer'
fast_track = 'Yes'
coop = 'Yes'
residency = 'International'
gender = 'Male'
prev_edu = 'Post Secondary'
age = '21 to 25'
first_term_gpa = 4.0
second_term_gpa = 2.0

input_arr = []
cat_list = [options_first_language, options_funding, options_fast_track, options_coop,
            options_residency, options_gender, options_prev_edu, options_age]

cat_array = [first_language, funding, fast_track, coop, residency, gender, prev_edu, age]
num_arr = [first_term_gpa, second_term_gpa]
for i in range(len(cat_list)):
    encoder = list(range(1, len(cat_list[i]) + 1))
    for j in range(len(cat_list[i])):
        if cat_list[i][j] == cat_array[i]:
            input_arr.append(encoder[j])
print(input_arr)

# pred_arr = np.array(num_arr + input_arr).reshape(1,-1)
# prediction = (model.predict(pred_arr) > 0.5).astype("int32")[0][0]
# print(prediction)

# print(pred_arr.shape)