import boto3

# AWS Configurations
role = "arn:aws:iam::<your-account-id>:role/<SageMaker-execution-role>"  # Update your IAM role
image_uri = "746669197371.dkr.ecr.us-east-2.amazonaws.com/sagemaker-custom-image:latest"
model_name = "music-captioning-model"
endpoint_config_name = "music-captioning-endpoint-config"
endpoint_name = "music-captioning-endpoint"
model_data_url = "s3://<your-s3-bucket>/model/model.tar.gz"  # Update with your S3 model path

# Initialize Boto3 client
sagemaker = boto3.client("sagemaker", region_name="us-east-2")

# 1. Create the model
try:
    response = sagemaker.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_data_url,
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",  # Update if your entry point differs
                "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",  # INFO
                "SAGEMAKER_REGION": "us-east-2",
            },
        },
        ExecutionRoleArn=role,
    )
    print(f"Model created: {response['ModelArn']}")
except Exception as e:
    print(f"Error creating model: {e}")

# 2. Create endpoint configuration
try:
    response = sagemaker.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InstanceType": "ml.m5.large",  # Change instance type if needed
                "InitialInstanceCount": 1,
            }
        ],
    )
    print(f"Endpoint config created: {response['EndpointConfigArn']}")
except Exception as e:
    print(f"Error creating endpoint config: {e}")

# 3. Create the endpoint
try:
    response = sagemaker.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name,
    )
    print(f"Endpoint creation initiated: {response['EndpointArn']}")
except Exception as e:
    print(f"Error creating endpoint: {e}")
