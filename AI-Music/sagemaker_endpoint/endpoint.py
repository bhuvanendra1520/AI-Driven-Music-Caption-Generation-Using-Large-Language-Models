import boto3

# SageMaker Runtime client
runtime_client = boto3.client("sagemaker-runtime", region_name="us-east-2")

# Input audio file path in S3
input_audio_s3 = "s3://<your-s3-bucket>/input/sample.mp3"

# Invoke the endpoint
response = runtime_client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="application/json",
    Body=f'{{"s3_uri": "{input_audio_s3}"}}',
)

# Decode the response
result = response["Body"].read().decode("utf-8")
print("Predictions:", result)
