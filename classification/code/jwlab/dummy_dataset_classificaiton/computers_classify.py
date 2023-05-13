from sktime.utils.load_data import load_from_tsfile_to_dataframe

train_x, train_y = load_from_tsfile_to_dataframe("./computers_dataset/Computers_TRAIN.ts", replace_missing_vals_with='NaN')
test_x, test_y = load_from_tsfile_to_dataframe("./computers_dataset/Computers_TEST.ts", replace_missing_vals_with='NaN')

