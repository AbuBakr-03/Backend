# APIBackend/views_questions.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404
from .models import Interview, Application
from .resume_analysis import ResumeAnalysisService
import logging

logger = logging.getLogger(__name__)


class InterviewQuestionsView(APIView):
    """
    API endpoint to generate interview questions from a resume
    """

    permission_classes = [IsAuthenticated]

    def post(self, request, pk=None):
        """
        Generate interview questions based on the resume
        """
        try:
            # Get the interview object
            interview_id = request.data.get("interview_id")
            if not interview_id and pk:
                interview_id = pk

            logger.info(f"Generating questions for interview_id: {interview_id}")

            interview = get_object_or_404(Interview, pk=interview_id)

            # Check permission - only recruiter or admin can generate questions
            user = request.user
            if not (user.is_staff or interview.application.job.recruiter == user):
                logger.warning(
                    f"User {user.username} tried to generate questions for interview {interview_id} without permission."
                )
                return Response(
                    {
                        "error": "You do not have permission to generate questions for this interview."
                    },
                    status=status.HTTP_403_FORBIDDEN,
                )

            # Check if resume exists
            if not interview.application.resume:
                logger.warning(
                    f"No resume file available for application {interview.application.id}."
                )
                return Response(
                    {"error": "No resume file found for this application."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Get the full path to the resume file
            resume_path = interview.application.resume.path

            # Get job description
            job = interview.application.job
            job_description = f"Title: {job.title}\nResponsibilities: {job.responsiblities}\nQualifications: {job.qualification}\nNice to haves: {job.nice_to_haves}"

            # Process the resume
            logger.info("Starting resume analysis for question generation.")
            analysis_service = ResumeAnalysisService()
            questions = analysis_service.analyze_resume_for_job(
                resume_path, job_description
            )

            if not questions:
                return Response(
                    {"error": "Failed to generate questions from resume."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

            # Update the interview with the generated questions
            interview.interview_questions = questions
            interview.save()

            return Response(
                {
                    "success": True,
                    "message": "Interview questions generated successfully.",
                    "questions": questions,
                }
            )

        except Exception as e:
            logger.error(f"Error generating interview questions: {e}")
            return Response(
                {"error": f"Failed to generate questions: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
