from pycaret.regression import *
import pandas as pd
import pickle
import gzip
import os
import boto3 

# Get the current directory of the file
current_dir = os.path.dirname(__file__)
print(current_dir)

def compress_pickle_gzip(file_path, data):
    with gzip.open(file_path, 'wb') as f:
        pickle.dump(data, f)

def main(df):
    s = setup(df, target='pm10_ref', session_id=123)

    # Compare baseline models
    best1 = compare_models()

    # Predict on test set
    holdout_pred1 = predict_model(best1)

    s3 = boto3.client('s3')
    s3_bucket_name = 'low-cost-air-sensor-calibration-engine'
    model_path_s3_key = 'assets/correction_factor_random_forest_sensor960-pm10-28-May-2024.pkl'
    compressed_model_path_s3_key = 'assets/model_pm10.pkl.gz'

    # Construct the full paths to the model file and compressed file
    model_save_path = os.path.join(current_dir, 'assets', 'correction_factor_random_forest_sensor960-pm10-28-May-2024')
    model_path = os.path.join(current_dir, 'assets', 'correction_factor_random_forest_sensor960-pm10-28-May-2024.pkl')
    compressed_file_path = os.path.join(current_dir, 'assets', 'model_pm10.pkl.gz')

    # Save the best model
    save_model(best1, model_save_path)

    # Save the best model to s3
    s3.upload_file(model_path, s3_bucket_name, model_path_s3_key)


    # Compress and save the model
    compress_pickle_gzip(compressed_file_path, model_path)

    # save Compressed model to s3
    s3.upload_file(compressed_file_path, s3_bucket_name, compressed_model_path_s3_key)


if __name__ == "__main__":
    
    # Construct the path to the CSV file
    csv_path = os.path.join(current_dir, 'assets', 'merged.csv')

    # Load the data
    df = pd.read_csv(csv_path, parse_dates=["DataDate"])

    # Filter the DataFrame
    filtered_df2 = df[["pm10_ref", "pm10", "temp", "hum"]]
    print(filtered_df2)

    # Run the main function
    main(filtered_df2)
