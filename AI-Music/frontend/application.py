from flask import Flask, request, jsonify, render_template
import boto3
import os

# Initialize Flask app
application = Flask(__name__)  # For Elastic Beanstalk compatibility

# AWS S3 Configuration
INPUT_BUCKET_NAME = os.getenv('INPUT_BUCKET_NAME', 'user-upload-audio-files')  # S3 bucket for uploads
CAPTIONS_BUCKET_NAME = os.getenv('CAPTIONS_BUCKET_NAME', 'generated-captions-storage')  # S3 bucket for captions
SUMMARY_BUCKET_NAME = os.getenv('SUMMARY_BUCKET_NAME', 'openai-generated-captions')  # S3 bucket for summaries

# Initialize the S3 client
s3 = boto3.client('s3')

@application.route('/')
def index():
    """
    Serves the main page of the web application.
    """
    return render_template('index.html'), 200

@application.route('/upload', methods=['POST'])
def upload_file_to_s3():
    """
    Uploads an MP3 file to the S3 'input/' folder.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.endswith('.mp3'):
        return jsonify({"error": "Only MP3 files are allowed"}), 400

    file_key = file.filename

    try:
        # Upload the file to the input bucket
        s3.upload_fileobj(file, INPUT_BUCKET_NAME, file_key)
        return jsonify({
            "message": "File uploaded successfully",
            "file_path": file_key,
            "file_name": file.filename
        }), 200
    except Exception as e:
        return jsonify({"error": f"Failed to upload file: {str(e)}"}), 500

@application.route('/get-captions-and-summary', methods=['GET'])
def get_captions_and_summary():
    """
    Fetches the captions and the OpenAI-generated summary based on the uploaded file's name.
    """
    file_name = request.args.get('file_name')

    if not file_name:
        return jsonify({"error": "No file name provided"}), 400

    base_file_name = file_name.rsplit('.mp3', 1)[0]
    captions_key = f"{base_file_name}_captions.txt"
    summary_key = f"{base_file_name}_captions_summary.txt"

    try:
        # Retrieve captions from the captions bucket
        captions_obj = s3.get_object(Bucket=CAPTIONS_BUCKET_NAME, Key=captions_key)
        captions = captions_obj['Body'].read().decode('utf-8')

        # Retrieve summary from the summary bucket
        summary_obj = s3.get_object(Bucket=SUMMARY_BUCKET_NAME, Key=summary_key)
        summary = summary_obj['Body'].read().decode('utf-8')

        return jsonify({
            "captions": captions,
            "summary": summary
        }), 200
    except s3.exceptions.NoSuchKey as e:
        error_msg = f"File not found: {e}"
        return jsonify({"error": error_msg}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve data: {str(e)}"}), 500

if __name__ == '__main__':
    # Use dynamic port from environment variable for Elastic Beanstalk
    # Run the Flask application in debug mode
    application.run(debug=True)
