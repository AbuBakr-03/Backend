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
    """Service for analyzing interview recordings with both audio and facial emotion recognition."""

    def __init__(self):
        # Get the base path to AImodels directory using Django settings
        self.base_path = os.path.join(settings.MODELS_ROOT)

        # Model paths using relative paths
        self.audio_model_path = os.path.join(
            self.base_path, "full_audio_emotion_model.h5"
        )
        self.face_model_path = os.path.join(self.base_path, "face_expression_model3.h5")
        self.scaler_path = os.path.join(self.base_path, "scaler2.pickle")
        self.encoder_path = os.path.join(self.base_path, "encoder2.pickle")

        # Haarcascade for face detection
        self.face_cascade_path = (
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Load models and preprocessing objects when needed
        self._audio_model = None
        self._face_model = None
        self._scaler = None
        self._encoder = None
        self._face_cascade = None

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

        # Direct mapping for face emotions (CV2 model uses different labels)
        self.face_emotion_labels = [
            "Angry",
            "Disgust",
            "Fear",
            "Happy",
            "Neutral",
            "Sad",
            "Surprise",
        ]

        # Emotion weights for confidence scoring
        self.emotion_weights = {
            "Happy": 1.00,  # Happiness is recognized most accurately
            "Surprise": 0.80,  # Second-highest accuracy
            "Neutral": 0.7,  # Emotional expressions are detected more accurately than neutral
            "Disgust": 0.55,  # Mid-tier accuracy (disgust > anger)
            "Angry": 0.4,  # Slightly below disgust (anger > sadness)
            "Sad": 0.3,  # Lower but above fear (sadness > fear)
            "Fear": 0.10,  # Fear is recognized least accurately
        }

        # Memory optimization settings
        self.chunk_duration = 5
        self.max_chunks = 30
        self.frame_sample_rate = 5  # Process 1 frame every N frames
        self.max_frames = 20  # Maximum number of frames to process

    def load_models(self):
        """Lazy-load models when needed"""
        # Load models with try-except blocks for better error handling
        try:
            # Configure GPU memory growth to prevent OOM errors
            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

            # Load audio model if not loaded
            if self._audio_model is None:
                logger.info(f"Loading audio model from {self.audio_model_path}")
                if os.path.exists(self.audio_model_path):
                    self._audio_model = load_model(self.audio_model_path)
                    logger.info("Audio emotion model loaded successfully")
                else:
                    logger.error(
                        f"Audio model file not found at {self.audio_model_path}"
                    )

            # Load face model if not loaded
            if self._face_model is None:
                logger.info(f"Loading face model from {self.face_model_path}")
                if os.path.exists(self.face_model_path):
                    self._face_model = load_model(self.face_model_path)
                    logger.info("Face emotion model loaded successfully")
                else:
                    logger.error(f"Face model file not found at {self.face_model_path}")

            # Load scaler if not loaded
            if self._scaler is None and os.path.exists(self.scaler_path):
                with open(self.scaler_path, "rb") as f:
                    self._scaler = pickle.load(f)
                logger.info("Scaler loaded successfully")

            # Load encoder if not loaded
            if self._encoder is None and os.path.exists(self.encoder_path):
                with open(self.encoder_path, "rb") as f:
                    self._encoder = pickle.load(f)
                logger.info("Encoder loaded successfully")

            # Load face cascade if not loaded
            if self._face_cascade is None:
                self._face_cascade = cv2.CascadeClassifier(self.face_cascade_path)
                logger.info("Face cascade classifier loaded successfully")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def process_recording(self, video_path):
        """
        Main function to process a video recording and analyze it for both
        facial expressions and audio emotions.

        Returns a dict with analysis results and a confidence score.
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

            # Analyze frames for facial emotions
            face_emotions = self.analyze_frames(frames_dir)

            # Combine emotions with appropriate weighting
            # Give more weight to facial expressions (70%) than audio (30%)
            combined_emotions = self.combine_emotions(audio_emotions, face_emotions)

            # Calculate confidence score
            confidence_score = self.calculate_confidence(combined_emotions)

            # Generate result
            result = self.determine_result(confidence_score)

            # Clean up
            shutil.rmtree(temp_dir)

            # Force garbage collection to free memory
            gc.collect()

            return {
                "emotions": dict(combined_emotions),
                "audio_emotions": dict(audio_emotions),
                "face_emotions": dict(face_emotions),
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
        """Extract audio from video file"""
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

    def extract_frames(self, video_path, output_dir):
        """Extract frames from video for facial analysis"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error("Error opening video file")
                return

            frame_count = 0
            saved_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Save frame at regular intervals
                if (
                    frame_count % self.frame_sample_rate == 0
                    and saved_count < self.max_frames
                ):
                    frame_path = os.path.join(
                        output_dir, f"frame_{saved_count:04d}.jpg"
                    )
                    cv2.imwrite(frame_path, frame)
                    saved_count += 1

                frame_count += 1

                # Stop if we've reached our max frames
                if saved_count >= self.max_frames:
                    break

            cap.release()
            logger.info(f"Extracted {saved_count} frames from video")

        except Exception as e:
            logger.error(f"Failed to extract frames: {e}")
            raise

    def analyze_audio(self, audio_dir):
        """Analyze audio clips for emotional content with memory efficiency"""
        # Load models if not already loaded
        self.load_models()

        # Check if audio model is available
        if self._audio_model is None:
            logger.warning("Audio model not available, skipping audio analysis")
            return Counter()

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

    def analyze_frames(self, frames_dir):
        """Analyze video frames for facial emotions"""
        # Load models if not already loaded
        self.load_models()

        # Check if face model is available
        if self._face_model is None or self._face_cascade is None:
            logger.warning("Face model not available, skipping face analysis")
            return Counter()

        # Get all image files
        image_files = [
            f
            for f in os.listdir(frames_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if not image_files:
            logger.warning("No image files found in the frames directory")
            return Counter()

        emotion_predictions = []

        for img_file in image_files:
            try:
                img_path = os.path.join(frames_dir, img_file)

                # Read and preprocess the image
                frame = cv2.imread(img_path)
                if frame is None:
                    continue

                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces
                faces = self._face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )

                # Process each face
                for x, y, w, h in faces:
                    face_roi = gray[y : y + h, x : x + w]

                    # Resize to match model input size
                    resized_face = cv2.resize(face_roi, (56, 56))

                    # Normalize
                    normalized_face = resized_face / 255.0

                    # Reshape for model
                    input_face = np.expand_dims(normalized_face, axis=0)
                    input_face = np.expand_dims(input_face, axis=-1)

                    # Predict emotion
                    prediction = self._face_model.predict(input_face, verbose=0)
                    emotion_idx = np.argmax(prediction)
                    emotion = self.face_emotion_labels[emotion_idx]
                    emotion_predictions.append(emotion)

            except Exception as e:
                logger.error(f"Error processing frame {img_file}: {e}")
                continue

        # Count occurrences of each emotion
        emotion_counts = Counter(emotion_predictions)
        return emotion_counts

    def combine_emotions(self, audio_emotions, face_emotions):
        """Combine audio and facial emotions with appropriate weighting"""
        # If either analysis is empty, use the other one
        if not audio_emotions:
            return face_emotions
        if not face_emotions:
            return audio_emotions

        # Combine with weighting (70% face, 30% audio)
        combined = Counter()

        # Normalize counters
        total_audio = sum(audio_emotions.values())
        total_face = sum(face_emotions.values())

        if total_audio > 0 and total_face > 0:
            # Add weighted audio emotions
            for emotion, count in audio_emotions.items():
                combined[emotion] += (count / total_audio) * 0.3

            # Add weighted face emotions
            for emotion, count in face_emotions.items():
                combined[emotion] += (count / total_face) * 0.7

            # Convert to integer counts
            total_samples = max(1, total_audio + total_face) // 2
            normalized = Counter()
            for emotion, weight in combined.items():
                normalized[emotion] = int(weight * total_samples)

            return normalized

        # If one is empty, return the other with full weight
        return face_emotions if total_face > 0 else audio_emotions

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
        if confidence_score >= 39:
            return 2  # Approved/Hired
        else:
            return 3  # Rejected
