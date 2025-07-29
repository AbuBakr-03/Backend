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
        Process an interview video and update the interview result - Optimized for Railway Pro
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

            # Memory optimization: Force garbage collection before starting
            import gc

            gc.collect()

            # Download video from R2 to temporary file with memory management
            import tempfile
            import requests

            video_url = interview.interview_video.url
            logger.info(f"Downloading video from R2: {video_url}")

            temp_video_path = None
            try:
                # Create temporary file with automatic cleanup
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp4"
                ) as temp_file:
                    # Download video in smaller chunks to manage memory better
                    response = requests.get(video_url, stream=True, timeout=300)
                    response.raise_for_status()

                    total_size = 0
                    chunk_size = 4096  # Smaller chunks for better memory management

                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            temp_file.write(chunk)
                            total_size += len(chunk)

                            # Memory check: Force GC every 1MB downloaded
                            if total_size % (1024 * 1024) == 0:
                                gc.collect()

                    temp_video_path = temp_file.name
                    logger.info(
                        f"Video downloaded ({total_size/(1024*1024):.2f}MB) to: {temp_video_path}"
                    )

                # Another GC before AI processing
                gc.collect()

                try:
                    # Process the video using your original AI analysis
                    logger.info("Starting AI analysis of the video...")

                    # Import tensorflow here to control memory growth
                    import tensorflow as tf

                    # Configure TensorFlow for better memory management
                    gpus = tf.config.experimental.list_physical_devices("GPU")
                    if gpus:
                        try:
                            for gpu in gpus:
                                tf.config.experimental.set_memory_growth(gpu, True)
                        except RuntimeError as e:
                            logger.warning(f"GPU config warning: {e}")

                    # Limit TensorFlow CPU memory usage
                    tf.config.threading.set_intra_op_parallelism_threads(2)
                    tf.config.threading.set_inter_op_parallelism_threads(2)

                    # Now run the analysis
                    analysis_service = InterviewAnalysisService()
                    analysis_result = analysis_service.process_recording(
                        temp_video_path
                    )

                    # Clean up memory after analysis
                    del analysis_service
                    gc.collect()

                    # Update the interview with the analysis results
                    logger.info(
                        f"Analysis completed. Updating interview with result: {analysis_result}"
                    )
                    interview.update_result_from_analysis(analysis_result)

                    if interview.result.id == 2:
                        from .models import PredictedCandidate

                        PredictedCandidate.objects.get_or_create(
                            interview=interview, defaults={"status_id": 1}
                        )

                    # Return the analysis results
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

                except Exception as e:
                    logger.error(f"Error processing video with AI: {e}")
                    import traceback

                    traceback.print_exc()
                    return Response(
                        {"error": f"AI analysis failed: {str(e)}"},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    )

            finally:
                # Clean up temporary file
                if temp_video_path and os.path.exists(temp_video_path):
                    try:
                        os.unlink(temp_video_path)
                        logger.info("Temporary video file cleaned up")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to cleanup temp file: {cleanup_error}")

                # Final garbage collection
                gc.collect()

        except Exception as e:
            logger.error(f"Unexpected error in InterviewRecordingView: {e}")
            import traceback

            traceback.print_exc()
            return Response(
                {"error": "An unexpected error occurred."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
