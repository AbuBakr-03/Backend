# APIBackend/views_interview.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404
from .models import Interview, Result
from .interview_analysis import InterviewAnalysisService
import os
import logging
import tempfile

logger = logging.getLogger(__name__)


class InterviewRecordingView(APIView):
    """
    API endpoint to handle interview video analysis
    """

    permission_classes = [IsAuthenticated]

    def post(self, request, pk=None):
        """
        Process an interview video and update the interview result - Slow but Real AI
        """
        try:
            # Get the interview object
            interview_id = request.data.get("interview_id")
            if not interview_id and pk:
                interview_id = pk

            logger.info(
                f"Processing interview analysis for interview_id: {interview_id}"
            )

            interview = get_object_or_404(Interview, pk=interview_id)

            # Check permission - only recruiter or admin can analyze recordings
            user = request.user
            if not (user.is_staff or interview.application.job.recruiter == user):
                logger.warning(
                    f"User {user.username} tried to analyze interview {interview_id} without permission."
                )
                return Response(
                    {"error": "You do not have permission to analyze this interview."},
                    status=status.HTTP_403_FORBIDDEN,
                )

            # Check if video exists
            if not interview.interview_video:
                logger.warning(f"No video file available for interview {interview_id}.")
                return Response(
                    {"error": "No video file found for this interview."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            logger.info(
                "Starting comprehensive AI analysis - this may take 60+ seconds..."
            )

            # STEP 1: Configure TensorFlow for maximum stability
            import os
            import gc

            # Force CPU-only mode for stability
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
            os.environ["OMP_NUM_THREADS"] = "2"

            import tensorflow as tf

            # Conservative TensorFlow configuration
            tf.config.set_soft_device_placement(True)
            tf.config.threading.set_intra_op_parallelism_threads(2)
            tf.config.threading.set_inter_op_parallelism_threads(2)

            # Disable GPU completely
            tf.config.set_visible_devices([], "GPU")

            logger.info("TensorFlow configured for CPU-only stable processing")

            # STEP 2: Download video with progress logging
            import tempfile
            import requests

            video_url = interview.interview_video.url
            logger.info(f"Downloading video from R2...")

            temp_video_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp4"
                ) as temp_file:
                    response = requests.get(
                        video_url, stream=True, timeout=600
                    )  # 10 min timeout
                    response.raise_for_status()

                    downloaded_size = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            temp_file.write(chunk)
                            downloaded_size += len(chunk)

                            # Log progress every MB
                            if downloaded_size % (1024 * 1024) == 0:
                                logger.info(
                                    f"Downloaded {downloaded_size // (1024*1024)}MB..."
                                )

                    temp_video_path = temp_file.name
                    logger.info(
                        f"Video download complete: {downloaded_size/(1024*1024):.2f}MB"
                    )

                # STEP 3: Clean memory before AI processing
                gc.collect()
                logger.info("Starting AI model initialization...")

                # STEP 4: Process with detailed logging
                try:
                    analysis_service = InterviewAnalysisService()
                    logger.info("InterviewAnalysisService created successfully")

                    # Load models with logging
                    logger.info("Loading AI models...")
                    analysis_service.load_models()
                    logger.info("AI models loaded successfully")

                    # Process the recording
                    logger.info("Starting video processing - this will take time...")
                    analysis_result = analysis_service.process_recording(
                        temp_video_path
                    )

                    logger.info(f"AI Analysis completed successfully!")
                    logger.info(f"Results: {analysis_result}")

                    # Clean up models from memory
                    del analysis_service
                    gc.collect()

                    # STEP 5: Update database with results
                    logger.info("Updating interview with AI analysis results...")
                    interview.update_result_from_analysis(analysis_result)

                    if interview.result.id == 2:
                        from .models import PredictedCandidate

                        PredictedCandidate.objects.get_or_create(
                            interview=interview, defaults={"status_id": 1}
                        )
                        logger.info("Predicted candidate record created")

                    logger.info(
                        f"Interview analysis complete - Result: {interview.result.title}"
                    )

                    return Response(
                        {
                            "success": True,
                            "message": f"Interview analyzed and result updated to: {interview.result.title}",
                            "emotions": analysis_result["emotions"],
                            "confidence": analysis_result["confidence"],
                            "result_id": interview.result.id,
                            "result_title": interview.result.title,
                        }
                    )

                except Exception as ai_error:
                    logger.error(f"AI processing failed: {ai_error}")
                    import traceback

                    traceback.print_exc()
                    raise ai_error

            finally:
                # Always clean up temporary file
                if temp_video_path and os.path.exists(temp_video_path):
                    try:
                        os.unlink(temp_video_path)
                        logger.info("Temporary video file cleaned up")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to cleanup temp file: {cleanup_error}")

                # Final cleanup
                gc.collect()

        except Exception as e:
            logger.error(f"Analysis failed with error: {e}")
            import traceback

            traceback.print_exc()
            return Response(
                {"error": f"Analysis failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
