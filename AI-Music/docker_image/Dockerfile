FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Install dependencies
RUN pip install transformers==4.30.2 torch torchvision torchaudio numpy
RUN pip install boto3

# Specify the working directory
WORKDIR /opt/ml/processing/code/