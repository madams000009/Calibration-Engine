from flask import Flask, request, jsonify
import pandas as pd
import joblib
import gzip
from flask_httpauth import HTTPBasicAuth
import os
from dotenv import load_dotenv
import pickle
import boto3

app = Flask(__name__)
auth = HTTPBasicAuth()

# Load environment variables from .env file
load_dotenv()

# Access the environment variables
user_singh = os.getenv('USER_SINGH')
user_adams = os.getenv('USER_ADAMS')
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
s3_bucket_name = os.getenv('S3_BUCKET_NAME')

# Check if the environment variables are set
if not user_singh or not user_adams or not aws_access_key_id or not aws_secret_access_key or not s3_bucket_name:
    raise EnvironmentError("Required environment variables are not set.")

# Dummy user data for demonstration
users = {
    "Singh": 'RR253675212LU',
    "Adams": 'Ad@m$05@080W)+]:'
}

@auth.verify_password
def verify_password(username, password):
    if username in users and users[username] == password:
        return username

def decompress_pickle_gzip(file_path):
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)

def download_file_from_s3(bucket_name, object_key, local_file_path):
    s3_client = boto3.client('s3',
                             aws_access_key_id=aws_access_key_id,
                             aws_secret_access_key=aws_secret_access_key)
    print(f"Downloading {object_key} from bucket {bucket_name} to {local_file_path}")
    s3_client.download_file(bucket_name, object_key, local_file_path)
    print(f"Download complete: {local_file_path}")

# Define the paths for the models in S3
model_pm2_5_s3_key = 'assets/model_pm2_5.pkl.gz'
model_pm10_s3_key = 'assets/model_pm10.pkl.gz'

# Define the local paths for the models
model_pm2_5_path = 'model_pm2_5.pkl.gz'
model_pm10_path = 'model_pm10.pkl.gz'

# Ensure the directory exists
os.makedirs('assets', exist_ok=True)

# Download the models from S3
download_file_from_s3(s3_bucket_name, model_pm2_5_s3_key, model_pm2_5_path)
download_file_from_s3(s3_bucket_name, model_pm10_s3_key, model_pm10_path)

# Load the models
model_pm2_5 = decompress_pickle_gzip(model_pm2_5_path)
model_pm2_5 = joblib.load(model_pm2_5)

model_pm10 = decompress_pickle_gzip(model_pm10_path)
model_pm10 = joblib.load(model_pm10)

@app.route('/')
def index():
    return 'Welcome to the Calibration Engine!'

@app.route('/calibration-engine/v1/', methods=['GET', 'POST'])
@auth.login_required
def predict_datapoints():
    if request.method == 'GET':
        return jsonify({'Instruction': 'Send JSON data with Hum, Temp, and PM2_5 for calibration'})

    # POST: Handle calibration prediction
    try:
        # Ensure there is JSON data and it is the correct format
        json_data = request.json
        if not json_data:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Convert JSON data to DataFrame
        if isinstance(json_data, list):
            json_to_df = pd.DataFrame(json_data)
        elif isinstance(json_data, dict):
            json_to_df = pd.DataFrame([json_data])
        else:
            return jsonify({'error': 'Unsupported JSON format'}), 400

        # Ensure required columns are present
        required_columns = ['hum', 'temp', 'pm2_5', 'pm10']
        for col in required_columns:
            if col not in json_to_df.columns:
                return jsonify({'error': f'Missing column: {col}'}), 400

        # Apply the formula to calculate Corrected pm2_5 and pm10
        filtered_df_pm2_5 = json_to_df[['hum', 'temp', 'pm2_5']]
        predictions_array_pm2_5 = model_pm2_5.predict(filtered_df_pm2_5)
        predictions_df_pm2_5 = pd.DataFrame(predictions_array_pm2_5, columns=['pm2_5'], index=json_to_df.index)

        filtered_df_pm10 = json_to_df[['hum', 'temp', 'pm10']]
        predictions_array_pm10 = model_pm10.predict(filtered_df_pm10)
        predictions_df_pm10 = pd.DataFrame(predictions_array_pm10, columns=['pm10'], index=json_to_df.index)

        json_to_df_dropped_pm2_5_pm10 = json_to_df.drop(columns=["pm2_5", "pm10"])

        combined_df = pd.concat([predictions_df_pm2_5, predictions_df_pm10, json_to_df_dropped_pm2_5_pm10], axis=1)

        return combined_df.to_json(orient='records')

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

#if __name__ == "__main__":
   # app.run(host="0.0.0.0", port=8080, debug=False)
