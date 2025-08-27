import openai
import boto3
import os

# Initialize S3 client
s3 = boto3.client('s3')

# Set OpenAI API Key (set this in Lambda's environment variables for security)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Fixed bucket names
INPUT_BUCKET_NAME = "generated-captions-storage"
OUTPUT_BUCKET_NAME = "openai-generated-captions"

def read_captions_from_s3(bucket_name, file_key):
    """
    Read captions from a text file stored in S3 and combine them into one string.
    """
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    lines = response['Body'].read().decode('utf-8').splitlines()
    # Combine all chunks into a single string, removing "Chunk X:" prefixes
    combined_captions = " ".join(line.split(":", 1)[1].strip() for line in lines if ":" in line)
    print("Combined Captions:", combined_captions)
    return combined_captions

def generate_summary(captions):
    """
    Send captions to OpenAI for summarization.
    """
    print("Sending captions to OpenAI for summarization...")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"These are the captions of a single audio clip at different time lapses. A single audio is divided into small chunks and fed to a model for chunk wise captions generation. Each chunk is of size 10 sec duration. So, each sentence corresponds to each chunk. The higher the number of sentences, the higher the duration of the audio clip. Based on this, summarize the following captions:\n{captions}"}
        ],
        max_tokens=250
    )
    summary = response['choices'][0]['message']['content'].strip()
    return summary

def save_summary_to_s3(bucket_name, file_key, summary):
    """
    Save the generated summary to a text file in S3.
    """
    s3.put_object(Bucket=bucket_name, Key=file_key, Body=summary)
    print(f"Summary saved to S3: s3://{bucket_name}/{file_key}")

def lambda_handler(event, context):
    """
    Lambda function entry point.
    """
    try:
        # Parse the event to get the file name
        record = event['Records'][0]
        input_key = record['s3']['object']['key']
        print(f"Received event for file: {input_key}")

        # Ensure the file is a .txt file
        if not input_key.endswith(".txt"):
            raise ValueError("Uploaded file is not a .txt file")

        # Step 1: Read captions from the input bucket
        captions = read_captions_from_s3(INPUT_BUCKET_NAME, input_key)
        print("Captions combined successfully.")

        # Step 2: Generate summary using OpenAI
        summary = generate_summary(captions)
        print("Summary generated successfully.")
        print("Generated Summary:")
        print(summary)

        # Step 3: Save the summary to the output bucket
        output_key = input_key.replace(".txt", "_summary.txt")
        save_summary_to_s3(OUTPUT_BUCKET_NAME, output_key, summary)

        return {
            "statusCode": 200,
            "body": {
                "message": "Summary generated and saved successfully.",
                "summary": summary,
                "output_file": f"s3://{OUTPUT_BUCKET_NAME}/{output_key}"
            }
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            "statusCode": 500,
            "body": {
                "message": "An error occurred.",
                "error": str(e)
            }
        }
