import os
import uuid
import tempfile
import shutil
import numpy as np
import cv2
import soundfile as sf
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import librosa
from sklearn.preprocessing import StandardScaler
from collections import Counter
import logging
from django.conf import settings

logger = logging.getLogger(__name__)


class InterviewAnalysisService:
    """
    Service to analyze interview recordings and determine candidate suitability.
    Uses both video facial expression analysis and audio emotion detection.
    """

    def __init__(self, model_path=None, scaler_path=None, encoder_path=None):
        """Initialize with optional paths to pre-trained models and preprocessing objects"""
        # Default paths - should be configured in settings
        self.base_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "AImodels"
        )

        # Model paths
        self.audio_model_path = "/home/abubakr/Backend/Backend/APIBackend/AImodels/full_audio_emotion_model.h5"

        self.scaler_path = (
            "/home/abubakr/Backend/Backend/APIBackend/AImodels/scaler2.pickle"
        )
        self.encoder_path = (
            "/home/abubakr/Backend/Backend/APIBackend/AImodels/encoder2.pickle"
        )

        # Load models and preprocessing objects when needed
        self._audio_model = None
        self._scaler = None
        self._encoder = None

        # Mapping for emotions to categories
        self.combined_mapping = {
            "angry": "Fear",  # Merged angry with fear
            "disgust": "Disgust",  # Leave as is
            "fear": "Fear",
            "happy": "Happy",
            "neutral": "Neutral",
            "sad": "Sad",
            "surprise": "Happy",  # Map surprise as Happy
        }

        # Emotion weights for confidence scoring
        self.emotion_weights = {
            "Happy": 1.00,  # Happiness is recognized most accurately
            "Surprise": 0.80,  # Second‐highest accuracy
            "Neutral": 0.7,  # Emotional expressions are detected more accurately than neutral
            "Disgust": 0.55,  # Mid‐tier accuracy (disgust > anger)
            "Angry": 0.5,  # Slightly below disgust (anger > sadness)
            "Sad": 0.45,  # Lower but above fear (sadness > fear)
            "Fear": 0.4,  # Fear is recognized least accurately
        }

    def load_models(self):
        """Lazy-load models when needed"""
        if self._audio_model is None:
            try:
                self._audio_model = load_model(self.audio_model_path)
                logger.info("Audio emotion model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load audio model: {e}")
                raise

        if self._scaler is None:
            try:
                with open(self.scaler_path, "rb") as f:
                    self._scaler = pickle.load(f)
                logger.info("Scaler loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load scaler: {e}")
                raise

        if self._encoder is None:
            try:
                with open(self.encoder_path, "rb") as f:
                    self._encoder = pickle.load(f)
                logger.info("Encoder loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load encoder: {e}")
                raise

    def process_recording(self, video_path):
        """
        Main function to process a video recording and analyze it
        Returns a dict with analysis results and a confidence score
        """
        try:
            # Create a temporary working directory
            temp_dir = tempfile.mkdtemp(prefix="interview_analysis_")

            # Extract audio and video frames
            audio_dir = os.path.join(temp_dir, "audio")
            frames_dir = os.path.join(temp_dir, "frames")
            os.makedirs(audio_dir, exist_ok=True)
            os.makedirs(frames_dir, exist_ok=True)

            # Extract audio from video
            audio_path = self.extract_audio(video_path, audio_dir)

            # Extract frames from video
            self.extract_frames(video_path, frames_dir)

            # Analyze audio for emotions
            audio_emotions = self.analyze_audio(audio_dir)

            # Analyze frames for facial expressions
            # Note: For this implementation we're focusing on audio analysis
            # video_emotions = self.analyze_frames(frames_dir)

            # Calculate confidence score
            confidence_score = self.calculate_confidence(audio_emotions)

            # Generate result
            result = self.determine_result(confidence_score)

            # Clean up
            shutil.rmtree(temp_dir)

            return {
                "emotions": dict(audio_emotions),
                "confidence": confidence_score,
                "result": result,
            }

        except Exception as e:
            logger.error(f"Failed to process recording: {e}")
            # Clean up if temp directory was created
            if "temp_dir" in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise

    def extract_audio(self, video_path, output_dir):
        """Extract audio from video file and split into clips"""
        try:
            # Base filename without extension
            base_name = os.path.splitext(os.path.basename(video_path))[0]

            # Full audio file path
            audio_path = os.path.join(output_dir, f"{base_name}.wav")

            # Use ffmpeg to extract audio
            os.system(f"ffmpeg -i {video_path} -q:a 0 -map a {audio_path}")

            # Load audio to split into clips
            data, sr = sf.read(audio_path)

            # Split audio into 5-second clips
            clip_duration = 5 * sr  # 5 seconds

            for i in range(0, len(data), clip_duration):
                clip = data[i : i + clip_duration]
                if len(clip) < sr:  # Skip clips shorter than 1 second
                    continue

                clip_path = os.path.join(
                    output_dir, f"{base_name}clip{i // clip_duration + 1}.wav"
                )
                sf.write(clip_path, clip, sr)

            return audio_path

        except Exception as e:
            logger.error(f"Failed to extract audio: {e}")
            raise

    def extract_frames(self, video_path, output_dir, frame_interval=90):
        """Extract frames from video at specified intervals"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")

            fps = cap.get(cv2.CAP_PROP_FPS)
            # Extract a frame every 3 seconds
            frame_step = int(fps * 3)

            frame_count = 0
            saved_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_step == 0:
                    frame_path = os.path.join(
                        output_dir, f"frame_{saved_count:04d}.jpg"
                    )
                    cv2.imwrite(frame_path, frame)
                    saved_count += 1

                frame_count += 1

            cap.release()
            return saved_count

        except Exception as e:
            logger.error(f"Failed to extract frames: {e}")
            raise

    def analyze_audio(self, audio_dir):
        """Analyze audio clips for emotional content"""
        # Load models if not already loaded
        self.load_models()

        # Get all WAV files
        wav_files = [f for f in os.listdir(audio_dir) if f.lower().endswith(".wav")]

        # Store emotion predictions
        emotion_predictions = []

        for wav_file in wav_files:
            file_path = os.path.join(audio_dir, wav_file)
            emotion = self._predict_audio_emotion(file_path)
            emotion_predictions.append(emotion)

        # Count occurrences of each emotion
        emotion_counts = Counter(emotion_predictions)

        return emotion_counts

    def _predict_audio_emotion(self, file_path):
        """Predict emotion from a single audio file"""
        try:
            # Extract features
            features = self._extract_audio_features(file_path)

            # Make prediction
            predictions = self._audio_model.predict(features)

            # Get label
            pred_index = np.argmax(predictions, axis=1)[0]
            raw_label = self._encoder.categories_[0][pred_index]

            # Map to final emotion category
            final_label = self.combined_mapping.get(raw_label, "Unknown")

            return final_label

        except Exception as e:
            logger.error(f"Error predicting audio emotion: {e}")
            return "Unknown"

    def _extract_audio_features(self, file_path, fixed_length=2376):
        """Extract and prepare audio features for prediction"""

        # Helper functions for feature extraction
        def zcr(data, frame_length, hop_length):
            zcr_val = librosa.feature.zero_crossing_rate(
                y=data, frame_length=frame_length, hop_length=hop_length
            )
            return np.squeeze(zcr_val)

        def rmse(data, frame_length=2048, hop_length=512):
            rmse_val = librosa.feature.rms(
                y=data, frame_length=frame_length, hop_length=hop_length
            )
            return np.squeeze(rmse_val)

        def mfcc(data, sr, frame_length=2048, hop_length=512, flatten=True):
            mfcc_val = librosa.feature.mfcc(y=data, sr=sr)
            return np.ravel(mfcc_val.T) if flatten else np.squeeze(mfcc_val.T)

        def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
            return np.hstack(
                (
                    zcr(data, frame_length, hop_length),
                    rmse(data, frame_length, hop_length),
                    mfcc(data, sr, frame_length, hop_length, flatten=True),
                )
            )

        # Load audio with fixed duration and offset
        data, sr = sf.read(file_path)

        # Extract features
        features = extract_features(data, sr)

        # Pad or truncate to fixed length
        current_length = len(features)
        if current_length < fixed_length:
            features = np.pad(
                features, (0, fixed_length - current_length), mode="constant"
            )
        elif current_length > fixed_length:
            features = features[:fixed_length]

        # Reshape for model input
        features = features.reshape(1, fixed_length)

        # Scale features
        scaled_features = self._scaler.transform(features)

        # Add channel dimension for CNN model
        model_input = np.expand_dims(scaled_features, axis=2)

        return model_input

    def calculate_confidence(self, emotion_counts):
        """Calculate a confidence score based on emotion distribution"""
        if not emotion_counts:
            return 0.0

        # Convert Counter to list of emotions
        emotions = []
        for emotion, count in emotion_counts.items():
            emotions.extend([emotion] * count)

        # Calculate score
        total_weight = sum(self.emotion_weights.get(e, 0.0) for e in emotions)
        confidence = total_weight / len(emotions) if emotions else 0.0

        return confidence * 100  # Return as percentage

    def determine_result(self, confidence_score):
        """
        Determine interview result based on confidence score
        Returns result ID (1 = pending, 2 = approved/hired, 3 = rejected)
        """
        # Threshold values can be adjusted
        if confidence_score >= 65:
            return 2  # Approved/Hired
        elif confidence_score <= 30:
            return 3  # Rejected
        else:
            return 1  # Keep as pending for human review
