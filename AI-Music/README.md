# Team-84: Bhuvanendra Putchala and Sneha Sasanapuri

# AI-Driven Music Caption Generation Using Large Language Models
<img width="842" alt="Screenshot 2024-12-13 at 9 37 09 PM" src="https://github.com/user-attachments/assets/81c2710e-cd67-4527-a094-92b9a689fe5a" />


## Introduction
This project provides an automated and scalable solution for generating natural language captions from audio files, leveraging pre-trained large language models (LLMs) such as BART. The system translates audio data into coherent captions, enabling applications in multimedia processing, accessibility, and media analysis.

Inspired by SeungHeon Doh's research, the project integrates AWS cloud services, pre-trained models, and advanced processing techniques. The goal is to streamline audio captioning tasks using an end-to-end pipeline, from audio upload to caption display.

---

## Architecture
The system integrates several AWS services and machine learning frameworks to achieve automation and scalability:

![image](https://github.com/user-attachments/assets/6b5b1252-67e9-4dad-bed0-db320c1a3840)

### Components:
1. **Web Application**:
   - Built using Flask and hosted on AWS Elastic Beanstalk (EBS).
   - Provides an interface for users to upload `.mp3` files and view captions.

2. **S3 Buckets**:
   - Store uploaded `.mp3` files, intermediate `.npy` files, and final captions.
     
3. **Lambda Functions**:
   - Handle audio file processing, SageMaker endpoint invocation, and OpenAI summarization API calls.

4. **SageMaker**:
   - Hosts the pre-trained BART model, which processes audio embeddings and generates captions.

5. **Audio Preprocessing**:
   - **FFmpeg**: Converts `.mp3` files into mel spectrograms.
   - Tensor representations of these spectrograms are processed for captioning.

6. **OpenAI GPT API**:
   - Summarizes chunk-wise captions into a cohesive description.

7. **CloudWatch**:
   - Monitors system performance and logs errors for debugging.

---

## How It Works

### Step-by-Step Workflow:
1. **User Upload**:
   - A user uploads a `.mp3` file through the web interface.
   - The file is stored in a designated S3 bucket.

2. **Audio Processing**:
   - An S3 event triggers a Lambda function to convert the `.mp3` file into `.npy` format.
   - The processed `.npy` file is stored in a second S3 bucket.

3. **Caption Generation**:
   - A second Lambda function invokes the SageMaker processing job, passing the `.npy` file as input.
   - SageMaker downloads the model artifacts, initializes the model, and generates chunk-wise captions.

4. **Caption Summarization**:
   - The generated captions are saved in an output bucket.
   - A third Lambda function triggers the OpenAI GPT API to summarize the captions into a unified description.

5. **Display Results**:
   - Both chunk-wise and summarized captions are displayed on the web interface.

---

## Deployment Instructions

If you want to deploy this project in AWS, follow the instructions below:

### Prerequisites:
1. **AWS Account**: Please make sure you can access an AWS account with sufficient permissions.
2. **CLI Tools**: Install the AWS CLI and configure it with your credentials.
3. **Docker**: Install Docker to build custom images.
4. **Codebase**: Clone the project repository from GitHub to your local machine.

### Deployment Steps:

1. **Set Up S3 Buckets**:
   - Create three S3 buckets:
     - **Input Bucket**: For storing uploaded `.mp3` files.
     - **Intermediate Bucket**: For storing `.npy` files.
     - **Output Bucket**: For storing generated captions.

2. **Launch Flask Web Application**:
   - Use AWS Elastic Beanstalk (EBS) to deploy the Flask application.
   - Create an EBS environment and upload your application code.
   - Configure the environment with IAM roles to access S3 buckets.

3. **Deploy Lambda Functions**:
   - Write three Lambda functions:
     - **Audio Processor**: Converts `.mp3` files to `.npy`.
     - **SageMaker Invoker**: Handles SageMaker processing job invocation.
     - **Summarizer**: Calls the OpenAI API to summarize captions.
   - Package the Lambda code and upload it to AWS Lambda.
   - Configure triggers for each Lambda function (S3 events and bucket notifications).

4. **Set Up SageMaker Endpoint**:
   - Use AWS CodeBuild to create a custom Docker image containing the BART model and dependencies.
   - Push the image to the Elastic Container Registry (ECR).
   - Create a SageMaker model using the ECR image and configure an endpoint.

5. **Configure CloudWatch Logs**:
   - Enable logging for all AWS services (SageMaker, Lambda, etc.) to monitor the system.
   - Use the logs to debug issues during deployment.

6. **Testing the Workflow**:
   - Upload a `.mp3` file via the web application.
   - Check the intermediate and output S3 buckets to verify the file processing and caption generation.
   - Validate the captions displayed on the web interface.

---

## Acknowledgment

This project is inspired by the work presented in the paper [LP-MusicCaps: LLM-Based Pseudo Music Captioning](https://arxiv.org/abs/2307.16372) by SeungHeon Doh, Keunwoo Choi, Jongpil Lee, and Juhan Nam. Their research provided valuable insights and foundational concepts that significantly contributed to the development of this project.
