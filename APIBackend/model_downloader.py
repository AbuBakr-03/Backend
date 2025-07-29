# APIBackend/model_downloader.py

# This file downloads your AI models from Cloudflare R2 storage to your server
# Think of it like downloading files from Google Drive to your computer

import os
import boto3
from django.conf import settings


def download_ai_models():
    """
    Simple function to download AI models from R2 cloud storage

    What this does:
    1. Connects to your Cloudflare R2 storage (where you uploaded the models)
    2. Creates a folder on the server to store the models
    3. Downloads each AI model file if it doesn't already exist
    4. Your Django app can then use these models for analysis
    """

    # Step 1: Connect to your Cloudflare R2 storage
    # This is like logging into your cloud storage account
    # We use the same credentials you already have for media files
    print("🔗 Connecting to Cloudflare R2 storage...")

    s3 = boto3.client(
        "s3",  # boto3 uses 's3' protocol even for Cloudflare R2
        endpoint_url=settings.CLOUDFLARE_R2_BUCKET_ENDPOINT,  # Your R2 server address
        aws_access_key_id=settings.CLOUDFLARE_R2_ACCESS_KEY,  # Your R2 username
        aws_secret_access_key=settings.CLOUDFLARE_R2_SECRET_KEY,  # Your R2 password
    )

    # Step 2: Create a folder on the server to store AI models
    # This is like creating a "Downloads" folder on your computer
    models_folder = settings.MODELS_ROOT  # Path where models will be saved
    os.makedirs(models_folder, exist_ok=True)  # Create folder if it doesn't exist
    print(f"📁 Models will be saved to: {models_folder}")

    # Step 3: List of your AI model files that need to be downloaded
    # These are the same files you uploaded to R2 in the previous step
    model_files = [
        "full_audio_emotion_model.h5",  # Analyzes emotions from audio
        "face_expression_model3.h5",  # Analyzes emotions from facial expressions
        "scaler2.pickle",  # Helper file for audio processing
        "encoder2.pickle",  # Helper file for audio processing
    ]

    print(f"📋 Need to download {len(model_files)} AI model files...")

    # Step 4: Download each model file one by one
    for model_file in model_files:
        # Where to save this file on the server
        local_file_path = os.path.join(models_folder, model_file)

        # Check if file already exists (no need to download again)
        if os.path.exists(local_file_path):
            print(f"✅ {model_file} already exists - skipping download")
            continue

        # Download the file from R2 cloud storage
        try:
            print(f"⬇️  Downloading {model_file} from cloud storage...")

            # This is like downloading a file from Google Drive
            s3.download_file(
                settings.CLOUDFLARE_R2_BUCKET,  # Your cloud storage bucket name
                f"ai-models/{model_file}",  # File location in cloud (ai-models folder)
                local_file_path,  # Where to save it on the server
            )

            # Check file size to confirm download worked
            file_size = os.path.getsize(local_file_path) / (1024 * 1024)  # Size in MB
            print(f"✅ Downloaded {model_file} ({file_size:.1f} MB)")

        except Exception as e:
            # If download fails, show error and stop
            print(f"❌ Failed to download {model_file}: {e}")
            print("   This might be due to network issues or incorrect file names")
            raise  # Stop the whole process if any model fails

    print("🎉 All AI models downloaded successfully!")
    print("   Your Django app can now use these models for interview analysis")


"""
FLOW EXPLANATION - What happens when this function runs:

1. 🔗 CONNECT TO CLOUD STORAGE
   - Your server connects to Cloudflare R2 (like logging into Google Drive)
   - Uses your existing R2 credentials from settings

2. 📁 PREPARE LOCAL STORAGE  
   - Creates a folder on the server to store AI models
   - Usually /tmp/models on Render servers

3. 📋 CHECK WHAT'S NEEDED
   - Lists the 4 AI model files that need to be downloaded
   - Each file serves a specific purpose in emotion analysis

4. ⬇️  DOWNLOAD PROCESS (for each file):
   - Check: Does this file already exist on server? 
   - If YES: Skip it (saves time and bandwidth)
   - If NO: Download it from cloud storage to server
   - Verify: Confirm download worked by checking file size

5. ✅ COMPLETION
   - All 4 AI models now exist on your server
   - Your InterviewAnalysisService can load and use them
   - Users can now upload videos and get AI analysis results

REAL WORLD ANALOGY:
Think of this like downloading apps to your phone:
- Your phone (server) needs certain apps (AI models) to work
- You download them once from the app store (R2 storage)  
- Once downloaded, you can use the apps anytime
- You don't need to download them again unless you delete them

THE RESULT:
After this runs, your server has all the AI models it needs to:
- Analyze facial expressions in interview videos
- Analyze emotions from audio
- Generate confidence scores
- Determine if candidates should be hired or rejected
"""
