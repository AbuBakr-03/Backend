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

            # Get the full path to the video file
            video_path = interview.interview_video.path

            try:
                # Process the video
                logger.info("Starting the analysis of the video.")
                analysis_service = InterviewAnalysisService()
                analysis_result = analysis_service.process_recording(video_path)

                # Update the interview with the analysis results
                logger.info(
                    f"Analysis completed. Updating interview with result: {analysis_result}"
                )
                interview.update_result_from_analysis(analysis_result)

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
                return Response(
                    {"error": f"Failed to process video: {str(e)}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

        except Exception as e:
            logger.error(f"Unexpected error in InterviewRecordingView: {e}")
            return Response(
                {"error": "An unexpected error occurred."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
