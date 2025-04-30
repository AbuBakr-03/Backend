#!/usr/bin/env python3
"""
Test script for interview analysis service.
This script tests the face and audio emotion detection functionality.

Usage:
    python test_emotion_detection.py /path/to/video.mp4

"""
import os
import sys
import django
import logging

# Set up Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BackendProject.settings")
django.setup()

from APIBackend.interview_analysis import InterviewAnalysisService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_interview_analysis(video_path):
    """Test the interview analysis service with a video file."""
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return

    logger.info(f"Testing interview analysis with video: {video_path}")

    try:
        # Initialize the service
        analysis_service = InterviewAnalysisService()

        # Process the video
        logger.info("Starting analysis...")
        result = analysis_service.process_recording(video_path)

        # Display results
        logger.info("Analysis completed successfully!")
        logger.info(f"Combined Emotions: {result['emotions']}")

        if "audio_emotions" in result:
            logger.info(f"Audio Emotions: {result['audio_emotions']}")

        if "face_emotions" in result:
            logger.info(f"Face Emotions: {result['face_emotions']}")

        logger.info(f"Confidence Score: {result['confidence']:.2f}%")
        logger.info(f"Result ID: {result['result']} (2=Pass, 3=Fail)")

        return result

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        return None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} /path/to/video.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    test_interview_analysis(video_path)
