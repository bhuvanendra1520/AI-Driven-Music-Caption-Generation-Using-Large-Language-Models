import boto3
import io
import numpy as np
import soundfile as sf
import subprocess
from typing import Tuple
from pathlib import Path

# AWS S3 Configuration
s3 = boto3.client('s3')
OUTPUT_BUCKET = "processes-in-between-storage"  # Replace with your output bucket name

# Constants for channel format
STR_CH_FIRST = 'channels_first'
STR_CH_LAST = 'channels_last'

def lambda_handler(event, context):
    """
    AWS Lambda entry point to process audio files from S3 and save as .npy.
    """
    # Extract bucket and file information from the event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    try:
        # Download the audio file from S3
        audio_data = download_from_s3(bucket, key)

        # Process the audio to generate a NumPy array
        audio_array = process_audio(audio_data)

        # Save the NumPy array as .npy in S3
        output_key = f"{key.split('.')[0]}.npy"
        upload_to_s3(audio_array, OUTPUT_BUCKET, output_key)

        return {"statusCode": 200, "body": f"Processed file saved as {output_key} in {OUTPUT_BUCKET}"}

    except Exception as e:
        return {"statusCode": 500, "body": f"Error processing file: {str(e)}"}


def download_from_s3(bucket: str, key: str) -> io.BytesIO:
    """
    Download a file from S3 and return its content as a BytesIO object.
    """
    response = s3.get_object(Bucket=bucket, Key=key)
    return io.BytesIO(response['Body'].read())


def upload_to_s3(array: np.ndarray, bucket: str, key: str):
    """
    Upload a NumPy array as .npy to S3.
    """
    buffer = io.BytesIO()
    np.save(buffer, array)  # Save array to buffer in .npy format
    buffer.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())


def process_audio(audio_data: io.BytesIO, duration=10, target_sr=16000) -> np.ndarray:
    """
    Process audio data and generate a NumPy array of chunks.

    Args:
        audio_data: Input audio data in bytes format.
        duration: Target duration of each chunk (in seconds).
        target_sr: Target sample rate.

    Returns:
        np.ndarray: Processed audio as a 2D NumPy array (chunks, n_samples).
    """
    # Save the file temporarily for FFmpeg processing
    temp_file = "/tmp/input_audio.mp3"
    with open(temp_file, "wb") as f:
        f.write(audio_data.read())
    n_samples = int(duration * target_sr)

    # Load and process the audio
    audio, sr = load_audio(
        path=temp_file,
        ch_format=STR_CH_FIRST,
        sample_rate=target_sr,
        downmix_to_mono=True,
        resample_by='ffmpeg',
    )

    if len(audio.shape) == 2:
        audio = audio.mean(0)  # to mono
    input_size = int(n_samples)
    if audio.shape[-1] < input_size:  # pad sequence
        pad = np.zeros(input_size)
        pad[: audio.shape[-1]] = audio
        audio = pad
    ceil = int(audio.shape[-1] // n_samples)
    audio_numpy = np.stack(np.split(audio[:ceil * n_samples], ceil)).astype('float32')
    return audio_numpy

    # ----------------

    # # Save the PyTorch tensor
    # torch.save(audio_tensor, "audio_tensor.pt")
    # print("PyTorch tensor saved to 'audio_tensor.pt'.")
    # ------------

def _resample_load_ffmpeg(path: str, sample_rate: int, downmix_to_mono: bool) -> Tuple[np.ndarray, int]:
    """
    Decode, downmix, and resample audio using FFmpeg.
    """
    channel_cmd = '-ac 1' if downmix_to_mono else ''
    resample_cmd = f'-ar {sample_rate}' if sample_rate else ''
    cmd = f"/opt/bin/ffmpeg -i {path} {channel_cmd} {resample_cmd} -f wav -"
    p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {err.decode('utf-8')}")
    return sf.read(io.BytesIO(out))


def load_audio(
    path: str or Path,
    ch_format: str,
    sample_rate: int = None,
    downmix_to_mono: bool = False,
    resample_by: str = 'ffmpeg',
    **kwargs,
) -> Tuple[np.ndarray, int]:
    """
    Load and optionally resample audio using FFmpeg.
    """
    if resample_by == 'ffmpeg':
        src, sr = _resample_load_ffmpeg(path, sample_rate, downmix_to_mono)
    else:
        raise ValueError(f"Unsupported resampling backend: {resample_by}")

    if ch_format == STR_CH_FIRST:
        src = np.expand_dims(src, axis=0)  # Convert to channels-first format
    return src, sr