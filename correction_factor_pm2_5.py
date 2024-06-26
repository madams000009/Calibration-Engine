from pycaret.regression import *
import pandas as pd
import pickle
import gzip
import os

# Get the current directory of the file
current_dir = os.path.dirname(__file__)
print(current_dir)

def compress_pickle_gzip(file_path, data):
    with gzip.open(file_path, 'wb') as f:
        pickle.dump(data, f)

def main(df):
    s = setup(df, target='pm2_5_ref', session_id=123)

    # Compare baseline models
    best1 = compare_models()

    # Predict on test set
    holdout_pred1 = predict_model(best1)

    # Construct the full paths to the model file and compressed file
    model_save_path = os.path.join(current_dir, 'assets', 'correction_factor_random_forest_sensor960-pm2_5-28-May-2024')
    model_path = os.path.join(current_dir, 'assets', 'correction_factor_random_forest_sensor960-pm2_5-28-May-2024.pkl')
    compressed_file_path = os.path.join(current_dir, 'assets', 'model_pm2_5.pkl.gz')

    # Save the best model
    save_model(best1, model_save_path)

    # Compress and save the model
    compress_pickle_gzip(compressed_file_path, model_path)

if __name__ == "__main__":
    # Construct the path to the CSV file
    csv_path = os.path.join(current_dir, 'assets', 'merged.csv')

    # Load the data
    df = pd.read_csv(csv_path, parse_dates=["DataDate"])

    # Filter the DataFrame
    filtered_df1 = df[["pm2_5_ref", "pm2_5", "temp", "hum"]]
    print(filtered_df1)

    # Run the main function
    main(filtered_df1)
