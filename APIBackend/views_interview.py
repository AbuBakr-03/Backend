# APIBackend/views_interview.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404
from .models import Interview, Result
from .interview_analysis import InterviewAnalysisService
import os
import tempfile
import logging
import base64
import uuid

logger = logging.getLogger(__name__)


class InterviewRecordingView(APIView):
    """
    API endpoint to handle candidate interview recordings
    """

    permission_classes = [IsAuthenticated]

    def post(self, request, pk=None):
        """
        Process an interview recording and update the interview result

        Expected request format:
        {
            "recording_data": "base64_encoded_webm_data",
            "interview_id": 123
        }
        """
        try:
            # Get the interview object
            interview_id = request.data.get("interview_id")
            if not interview_id and pk:
                interview_id = pk

            logger.info(
                f"Processing interview recording for interview_id: {interview_id}"
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

            # Get recording data
            recording_data = request.data.get("recording_data")
            if not recording_data:
                logger.warning(
                    f"No recording data provided for interview {interview_id}."
                )
                return Response(
                    {"error": "No recording data provided."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Decode base64 data
            try:
                # Remove potential data URL prefix
                if "base64," in recording_data:
                    recording_data = recording_data.split("base64,")[1]

                logger.info("Decoding base64 recording data.")
                decoded_data = base64.b64decode(recording_data)
                logger.info("Base64 data decoded successfully.")

            except Exception as e:
                logger.error(f"Failed to decode base64 data: {e}")
                return Response(
                    {"error": "Invalid recording data format."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Create a temporary file for the recording
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".webm", delete=False
                ) as temp_file:
                    temp_file.write(decoded_data)
                    temp_file_path = temp_file.name

                logger.info(f"Temporary file created for recording: {temp_file_path}")

            except Exception as e:
                logger.error(f"Error creating temporary file for recording: {e}")
                return Response(
                    {"error": "Failed to create temporary file for recording."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

            try:
                # Process the recording
                logger.info("Starting the analysis of the recording.")
                analysis_service = InterviewAnalysisService()
                analysis_result = analysis_service.process_recording(temp_file_path)

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
                logger.error(f"Error processing recording: {e}")
                return Response(
                    {"error": f"Failed to process recording: {str(e)}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

            finally:
                # Clean up the temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    logger.info(
                        f"Temporary file {temp_file_path} removed after processing."
                    )

        except Exception as e:
            logger.error(f"Unexpected error in InterviewRecordingView: {e}")
            return Response(
                {"error": "An unexpected error occurred."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
