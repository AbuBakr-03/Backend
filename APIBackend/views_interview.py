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
        Process an interview video and update the interview result
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

            # RAILWAY FIX: Download video from R2 to temporary file
            import tempfile
            import requests

            video_url = interview.interview_video.url
            logger.info(f"Downloading video from R2: {video_url}")

            # Create temporary file for analysis
            temp_video_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp4"
                ) as temp_file:
                    # Download video in chunks to manage memory
                    response = requests.get(video_url, stream=True)
                    response.raise_for_status()

                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            temp_file.write(chunk)

                    temp_video_path = temp_file.name
                    logger.info(
                        f"Video downloaded to temporary file: {temp_video_path}"
                    )

                try:
                    # Process the video using your original AI analysis
                    logger.info("Starting the analysis of the video.")
                    analysis_service = InterviewAnalysisService()
                    analysis_result = analysis_service.process_recording(
                        temp_video_path
                    )

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
                    logger.error(f"Error processing video: {e}")
                    import traceback

                    traceback.print_exc()
                    return Response(
                        {"error": f"Failed to process video: {str(e)}"},
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

        except Exception as e:
            logger.error(f"Unexpected error in InterviewRecordingView: {e}")
            import traceback

            traceback.print_exc()
            return Response(
                {"error": "An unexpected error occurred."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
