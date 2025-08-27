import boto3
import json
import time

# Initialize SageMaker and S3 clients
sagemaker = boto3.client('sagemaker')

def lambda_handler(event, context):
    # Parse the S3 event to get the uploaded file details
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    file_key = event['Records'][0]['s3']['object']['key']
    
    print(f"Triggered by file: s3://{bucket_name}/{file_key}")

    # Define paths for SageMaker Processing Job
    input_s3_uri = f"s3://{bucket_name}/{file_key}"  # Path of the uploaded .npy file
    model_s3_uri = "s3://lp-music-caps-model/model-code/model.tar.gz"  # Model archive
    output_s3_uri = "s3://generated-captions-storage/"  # Output path for captions

    # Generate a unique processing job name
    timestamp = int(time.time())
    processing_job_name = f"audio-captioning-job-{timestamp}"

    # Start the SageMaker Processing Job
    response = sagemaker.create_processing_job(
        ProcessingJobName=processing_job_name,
        RoleArn="arn:aws:iam::058264481125:role/sagemaker-execution-role",  # Replace with your role ARN
        AppSpecification={
            "ImageUri": "058264481125.dkr.ecr.us-east-2.amazonaws.com/sagemaker-processing-image:latest",
            "ContainerEntrypoint": ["python3", "/opt/ml/processing/code/process_audio.py"],
            "ContainerArguments": [file_key]  # Pass FILE_KEY as a command-line argument
        },
        ProcessingInputs=[
            {
                "InputName": "input-audio",
                "S3Input": {
                    "S3Uri": input_s3_uri,
                    "LocalPath": "/opt/ml/processing/input",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File"
                }
            },
            {
                "InputName": "model-archive",
                "S3Input": {
                    "S3Uri": model_s3_uri,
                    "LocalPath": "/opt/ml/processing/model",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File"
                }
            }
        ],
        ProcessingOutputConfig={
            "Outputs": [
                {
                    "OutputName": "captions",
                    "S3Output": {
                        "S3Uri": output_s3_uri,
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob"
                    }
                }
            ]
        },
        ProcessingResources={
            "ClusterConfig": {
                "InstanceType": "ml.m5.large",
                "InstanceCount": 1,
                "VolumeSizeInGB": 30
            }
        },
        StoppingCondition={
            "MaxRuntimeInSeconds": 1800
        }
    )

    print(f"Started Processing Job: {response['ProcessingJobArn']}")
    
    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "SageMaker Processing Job started successfully.",
            "processing_job_name": processing_job_name
        })
    }
