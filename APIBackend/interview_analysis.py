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
import gc  
logger = logging.getLogger(__name__)

class InterviewAnalysisService:
    

    def __init__(self, model_path=None, scaler_path=None, encoder_path=None):
        
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
            "angry": "Fear",  
            "disgust": "Disgust",  
            "fear": "Fear",
            "happy": "Happy",
            "neutral": "Neutral",
            "sad": "Sad",
            "surprise": "Happy",  
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

        # Memory optimization settings
        self.chunk_duration = 5 
        self.max_chunks = (
            30  
        )

    def load_models(self):
        """Lazy-load models when needed"""
        if self._audio_model is None:
            try:
                gpus = tf.config.experimental.list_physical_devices("GPU")
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)

                # Load model with reduced precision to save memory
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
            os.makedirs(audio_dir, exist_ok=True)

            # Extract audio from video
            audio_path = self.extract_audio(video_path, audio_dir)

            # Analyze audio for emotions
            audio_emotions = self.analyze_audio(audio_dir)

            # Calculate confidence score
            confidence_score = self.calculate_confidence(audio_emotions)

            # Generate result
            result = self.determine_result(confidence_score)

            # Clean up
            shutil.rmtree(temp_dir)

            # Force garbage collection to free memory
            gc.collect()

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

            # Force garbage collection
            gc.collect()

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

            return audio_path

        except Exception as e:
            logger.error(f"Failed to extract audio: {e}")
            raise

    def analyze_audio(self, audio_dir):
        """Analyze audio clips for emotional content with memory efficiency"""
        # Load models if not already loaded
        self.load_models()

        # Get all WAV files
        wav_files = [f for f in os.listdir(audio_dir) if f.lower().endswith(".wav")]
        if not wav_files:
            logger.warning("No WAV files found in the audio directory")
            return Counter()

        # Use the first WAV file (there should be only one main audio file)
        file_path = os.path.join(audio_dir, wav_files[0])

        try:
            # Process audio in chunks to save memory
            emotion_predictions = self._predict_audio_emotion_chunked(file_path)

            # Count occurrences of each emotion
            emotion_counts = Counter(emotion_predictions)

            return emotion_counts

        except Exception as e:
            logger.error(f"Error analyzing audio: {e}")
            return Counter()

    def _predict_audio_emotion_chunked(self, file_path):
        """Predict emotion from audio file in chunks to reduce memory usage"""
        try:
            # Load audio file info without loading all data
            audio_info = sf.info(file_path)
            sample_rate = audio_info.samplerate
            total_samples = audio_info.frames
            channels = audio_info.channels

            # Calculate chunk sizes in samples
            chunk_samples = int(sample_rate * self.chunk_duration)

            # Determine number of chunks
            num_chunks = min(self.max_chunks, total_samples // chunk_samples)

            logger.info(
                f"Processing {num_chunks} audio chunks of {self.chunk_duration}s each"
            )

            # Process audio in chunks
            emotion_predictions = []

            for i in range(num_chunks):
                start_sample = i * chunk_samples

                # Read just this chunk of audio
                with sf.SoundFile(file_path, "r") as f:
                    f.seek(start_sample)
                    chunk_data = f.read(chunk_samples)

                # Mono conversion if stereo
                if channels > 1:
                    chunk_data = np.mean(chunk_data, axis=1)

                # Extract features for this chunk
                features = self._extract_audio_features_optimized(
                    chunk_data, sample_rate
                )

                # Make prediction
                if features is not None:
                    predictions = self._audio_model.predict(features, verbose=0)
                    pred_index = np.argmax(predictions, axis=1)[0]
                    raw_label = self._encoder.categories_[0][pred_index]

                    # Map to final emotion category
                    final_label = self.combined_mapping.get(raw_label, "Neutral")
                    emotion_predictions.append(final_label)

                # Clear memory
                del chunk_data
                if features is not None:
                    del features
                gc.collect()

            return emotion_predictions

        except Exception as e:
            logger.error(f"Error predicting audio emotion in chunks: {e}")
            return ["Neutral"]  # Default to neutral on error

    def _extract_audio_features_optimized(self, data, sr, fixed_length=2376):
        """Extract audio features with memory optimization"""
        try:
            # Calculate basic features with low memory usage
            # For Zero Crossing Rate (ZCR)
            zcr = librosa.feature.zero_crossing_rate(
                y=data, frame_length=2048, hop_length=512
            )
            zcr = np.squeeze(zcr)

            # For Root Mean Square Energy (RMSE)
            rmse = librosa.feature.rms(y=data, frame_length=2048, hop_length=512)
            rmse = np.squeeze(rmse)

            # For MFCC features - use fewer coefficients to save memory
            mfcc_feat = librosa.feature.mfcc(
                y=data, sr=sr, n_mfcc=13
            )  # Reduced from default
            mfcc_feat = np.ravel(mfcc_feat.T)

            # Combine features
            features = np.hstack((zcr, rmse, mfcc_feat))

            # Handle length issues
            current_length = len(features)
            if current_length < fixed_length:
                # Pad if too short
                features = np.pad(
                    features, (0, fixed_length - current_length), mode="constant"
                )
            elif current_length > fixed_length:
                # Truncate if too long
                features = features[:fixed_length]

            # Reshape and process for model
            features = features.reshape(1, fixed_length)
            scaled_features = self._scaler.transform(features)
            model_input = np.expand_dims(scaled_features, axis=2)

            return model_input

        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            return None

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
        if confidence_score >= 50:
            return 2  # Approved/Hired

        else:
            return 3  # Keep as pending for human review
