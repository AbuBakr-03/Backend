# upload_models.py - Run this once locally to upload your models to R2
import os
import boto3
from decouple import config
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def upload_models():
    # Your existing R2 credentials from settings
    s3_client = boto3.client(
        "s3",
        endpoint_url=config("CLOUDFLARE_R2_BUCKET_ENDPOINT"),
        aws_access_key_id=config("CLOUDFLARE_R2_ACCESS_KEY"),
        aws_secret_access_key=config("CLOUDFLARE_R2_SECRET_KEY"),
    )

    bucket_name = config("CLOUDFLARE_R2_BUCKET")

    # Your local models directory
    local_models_dir = "APIBackend/AImodels"

    # Models to upload
    models_to_upload = [
        "full_audio_emotion_model.h5",
        "face_expression_model3.h5",
        "scaler2.pickle",
        "encoder2.pickle",
    ]

    print("Starting model upload to Cloudflare R2...")

    for model_file in models_to_upload:
        local_path = os.path.join(local_models_dir, model_file)

        if os.path.exists(local_path):
            # Upload to ai-models/ folder in R2
            r2_key = f"ai-models/{model_file}"

            try:
                print(f"Uploading {model_file}...")
                file_size = os.path.getsize(local_path)
                print(f"  File size: {file_size / (1024*1024):.1f} MB")

                s3_client.upload_file(local_path, bucket_name, r2_key)
                print(f"  ✅ Successfully uploaded {model_file}")

            except Exception as e:
                print(f"  ❌ Failed to upload {model_file}: {e}")
        else:
            print(f"  ⚠️  File not found: {local_path}")

    print("\nUpload complete!")
    print(
        "\nUploaded files should be visible in your R2 bucket under 'ai-models/' folder"
    )


if __name__ == "__main__":
    upload_models()
