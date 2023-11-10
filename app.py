import streamlit as st
import boto3
import pandas as pd
import os
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.deserializers import CSVDeserializer #JSONDeserializer
from sagemaker.serializers import CSVSerializer
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, \
                            ConfusionMatrixDisplay, precision_score, recall_score, \
                            f1_score, roc_curve, roc_auc_score
from time import sleep
from matplotlib import pyplot as plt
from sagemaker.model_monitor import ModelQualityMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat
from datetime import datetime

s3_bucket = 'sagemaker-studio-073855787749-u265vam8zvs'
data_key = 'input/On-time-delivery-data.csv'
data_location = f's3://{s3_bucket}/{data_key}'
s3_client = boto3.client('s3')
role = 'arn:aws:iam::073855787749:role/service-role/AmazonSageMaker-ExecutionRole-20231027T121566'

sagemaker_session = sagemaker.Session()
# Define the SageMaker endpoint
sage_client = boto3.client('sagemaker-runtime', region_name=sagemaker_session.boto_region_name)
endpoint_name = 'DEMO-xgboost-2023-11-08-1708'

predictor = Predictor(endpoint_name=endpoint_name,
                      serializer = CSVSerializer(),
                      deserializer = CSVDeserializer())



model_quality_monitor = ModelQualityMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.m4.xlarge',
    volume_size_in_gb=16,
    max_runtime_in_seconds=1800,
    sagemaker_session=sagemaker_session
)

# Load the scaler_obj
scaler_obj = joblib.load('scaler_obj.pkl')

# Load the columns_to_remove
columns_to_remove = joblib.load('columns_to_remove.pkl')
text_X= pd.read_csv('test_X.csv')
def group_zipcodes(zipcode):
    # Use the first 3 digits to group zip codes
    return str(zipcode)[:1]

def preprocess(df, typedf='new_point', scaler=scaler_obj, selection=False, columns_to_remove = columns_to_remove):
  
    df_new = df.loc[:, [col for col in df.columns if col not in columns_to_remove]]
    # grouping zipcode to reduce cardinality

    # Apply the grouping function to the 'zipcode' column
    df_new['zipcode_grouped'] = df_new['zipcode'].apply(group_zipcodes)
    df_new = df_new.loc[:, df_new.columns != 'zipcode']
    
    if selection:
        df_new = df_new.loc[:, [col for col in df_new.columns if col not in columns_to_remove]]
        
    
    # Initialize the StandardScaler
    categorical_columns = ['classification_ontime','zipcode_grouped']
    numeric_columns = [col for col in df_new.columns if col not in categorical_columns]
    
    # Fit the StandardScaler on the training data and transform the training data
    df_std = df_new.copy()
    df_std[numeric_columns] = scaler.transform(df_new[numeric_columns])
    
    # onehotencoding
    for value in range(1,10):
        df_std['zipcode_grouped_' + str(value)] = (df_std['zipcode_grouped'] == value).astype(int)
    df_std = df_std.drop('zipcode_grouped', axis = 1)
    df_std = df_std.reset_index(drop=True)
    if typedf=='new_point':
        return df_std
    else:
        df_std_X = df_std.drop('classification_ontime', axis=1)
        df_std['classification_ontime'] = df_std['classification_ontime'].apply(lambda x: 1 if x=='On time' else 0)
        df_std_Y = df_std['classification_ontime']
        df_std_X = df_std_X.reset_index(drop=True)
        df_std_Y = df_std_Y.reset_index(drop=True)
        return df_std_X, df_std_Y

# Function to make predictions using SageMaker endpoint
def predict(input_data):
    probability = float(predictor.predict(input_data)[0][0])
    prediction = "On time" if probability > 0.5 else "Delayed"
    return prediction, float(probability)

def predict_batch(input_data, limit=200):
    progress_bar = st.progress(0)
    test_X, test_Y= preprocess(input_data, typedf='new_batch', scaler=scaler_obj, selection=True, columns_to_remove=columns_to_remove)
    i=0
    test_Y_pred = pd.Series([])
    for index, row in test_X.iterrows():
        if i==limit:
            break
        probability = float(predictor.predict(row)[0][0])
        prediction = 1 if probability > 0.5 else 0
        test_Y_pred = pd.concat([test_Y_pred, pd.Series(prediction)])
        i+=1
        progress_bar.progress(int(100 * i / limit))
        print(".", end="", flush=True)
        sleep(0.01)
    test_Y_limit = test_Y_pred.iloc[0:limit]
    print("Done!")
    accuracy = accuracy_score(test_Y_limit, test_Y_pred)
    precision = precision_score(test_Y_limit, test_Y_pred)
    recall = recall_score(test_Y_limit, test_Y_pred)
    f1 = f1_score(test_Y_limit, test_Y_pred)
    st.subheader("Metrics")
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(test_Y_limit, test_Y_pred)
    roc_auc = roc_auc_score(test_Y_limit, test_Y_pred)
    st.write("Validation Accuracy: {:.2f}".format(accuracy))
    st.write("Precision: {:.2f}".format(precision))
    st.write("Recall: {:.2f}".format(recall))
    st.write("F1 Score: {:.2f}".format(f1))
    st.write("AUC: {:.2f}".format(roc_auc))


    # Create and display a modern confusion matrix
    cm = confusion_matrix(test_Y_limit, test_Y_pred)
    #disp = plot_confusion_matrix(best_xgb, val_X, val_Y, display_labels=, cmap=plt.cm.Blues, values_format='d')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Delayed', 'On time'])
    disp.plot()
    st.write("Confusion Matrix")
    st.pyplot(plt)


    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    st.pyplot(plt)

def get_metrics(df, limit=500):
    # progress_bar = st.progress(0)
    # train_dataset = "train_with_predictions.csv"
    # train_X, train_Y= preprocess(df, typedf='new_batch', scaler=scaler_obj, selection=True, columns_to_remove=columns_to_remove)
    # i=0
    # with open(f"{train_dataset}", "w") as validation_file:
    #     validation_file.write("probability,prediction,label\n")  # CSV header
    #     for index, row in train_X.iterrows():
    #         if i==limit:
    #             break
    #         label = train_Y.iloc[index]
    #         probability = float(predictor.predict(row)[0][0])
    #         prediction = "1" if probability > 0.5 else "0"
    #         validation_file.write(f"{probability},{prediction},{label}\n")
    #         progress_bar.progress(int(100 * i / limit)+1)
    #         i += 1

    #         print(".", end="", flush=True)
    #         sleep(0.02)
    # print("Done!")

    # s3_client.upload_file(Filename=train_dataset, Bucket=s3_bucket, Key=train_dataset)
    # sleep(3)
    # baseline_job_name = f"MyBaseLineJob-{datetime.utcnow():%Y-%m-%d-%H%M}"
    # baseline_dataset_uri = 's3://' + s3_bucket +'/' +train_dataset #s3://sagemaker-studio-073855787749-u265vam8zvs/test_with_predictions.csv

    # baseline_results_uri = os.path.join('s3://', s3_bucket, 'output')
    # job = model_quality_monitor.suggest_baseline(
    #     job_name=baseline_job_name,
    #     baseline_dataset=baseline_dataset_uri, # The S3 location of the validation dataset.
    #     dataset_format=DatasetFormat.csv(header=True),
    #     output_s3_uri = baseline_results_uri, # The S3 location to store the results.
    #     problem_type='BinaryClassification',
    #     inference_attribute= "prediction", # The column in the dataset that contains predictions.
    #     probability_attribute= "probability", # The column in the dataset that contains probabilities.
    #     ground_truth_attribute= "label" # The column in the dataset that contains ground truth labels.
    # )
    # job.wait(logs=False)
    
    #baseline_job = model_quality_monitor.latest_baselining_job
    # print(baseline_job)
    # print(model_quality_monitor)
    baseline_job = model_quality_monitor.describe_baseline_job(job_name='MyBaseLineJob-2023-11-08-1754')
    st.write(pd.DataFrame(baseline_job.suggested_constraints().body_dict["binary_classification_constraints"]).T)

# Streamlit app
st.title("On time or Delayed App")
st.header('Model Prediction')
use_sample_input = st.checkbox("Use Sample Input")
with st.expander("Click to check data"):
    if use_sample_input:
        input_data = text_X.sample(n=1)
        st.subheader("Input Sample Data:")
        st.write(input_data)
    else:

        # Input form
        st.header("Enter Input Data")
        # Create an empty DataFrame with the specified columns
        columns = [
            'zipcode', 'total_items', 'precipitation_rate', 'water_runoff', 'snow_depth',
            'temperature', 'temperature_at_1500m', 'min_temperature', 'max_temperature',
            'pressure', 'wind_gust_speed', 'total_cloud_cover', 'dew_point_temperature',
            'relative_humidity', 'wind_speed'
        ]
        data = pd.DataFrame(columns=columns)


        # Create input fields for each column
        for column in columns:
            if column == 'zipcode':
                input_value = st.number_input(f"Enter {column}", step=1)
            else:
                input_value = st.number_input(f"Enter {column}")
            data[column] = [input_value]

        # Display the input data
        st.subheader("Input Data:")
        st.write(data)
        # test_X = pd.read_csv('test_X.csv')
        # print(test_X.dtypes)

        input_data = preprocess(data, typedf='new_point', scaler=scaler_obj, selection=True, columns_to_remove=columns_to_remove).iloc[0,:].astype(float)
        #input_data = test_X.iloc[1,:]
        #print(input_data.dtypes)

# Make predictions and display results

if st.button('Predict',  key=1):
    result, prob= predict(input_data)
    st.subheader("Prediction Probability")
    st.write(prob)

    st.subheader("Classification Result")
    st.write(result)

st.header('Model Evaluation')
if st.checkbox("Predict Test set"):
    uploaded_file_1 = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file_1 is not None:
        #try:
        limit_1 = st.number_input('Enter number of rows', step=1)
        if st.button('Predict', key=2):
            test_data = pd.read_csv(uploaded_file_1)
            predict_batch(test_data, limit=limit_1)
        # except Exception as e:
        #     st.write("Invalid file.")

# st.header('Model Monitoring')
# if st.checkbox("Get Quality Metrics"):
#     uploaded_file_2 = st.file_uploader("Upload a CSV file", type=["csv"], key=2)
#     if uploaded_file_2 is not None:
#         #try:
#         limit_2 = st.number_input('Enter number of rows', step=1)
#         if st.button('Get Metrics', key=3):
#             train_data = pd.read_csv(uploaded_file_2)
#             get_metrics(train_data, limit=limit_2)
#         # except Exception as e:
#         #     st.write("Invalid file.")

