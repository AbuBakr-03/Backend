import tempfile
import os
from django.shortcuts import get_object_or_404
from django.core.files.storage import default_storage
from rest_framework import generics, status
from rest_framework.views import APIView
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import (
    IsAuthenticated,
    IsAdminUser,
    BasePermission,
    AllowAny,
)
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.filters import SearchFilter, OrderingFilter
from django_filters.rest_framework import DjangoFilterBackend

from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

from .serializers import (
    DepartmentSerializer,
    CompanySerializer,
    JobSerializer,
    StatusSerializer,
    ApplicationSerializer,
    ResultSerializer,
    InterviewSerializer,
    PredictedCandidateSerializer,
    CustomTokenObtainPairSerializer,
    CustomTokenRefreshSerializer,
)

from .models import (
    Department,
    Company,
    Job,
    Status,
    Application,
    Result,
    Interview,
    PredictedCandidate,
)

from .services import (
    ResumeScreeningService,
    CandidateEvaluationService,
)

from .interview_analysis import InterviewAnalysisService

import os
from BackendProject import settings


class CustomTokenObtainPairView(TokenObtainPairView):
    serializer_class = CustomTokenObtainPairSerializer


class isRecruiter(BasePermission):
    def has_permission(self, request, view):
        return request.user and (
            request.user.is_staff
            or request.user.groups.filter(name="Recruiter").exists()
        )


class DepartmentView(generics.ListCreateAPIView):
    queryset = Department.objects.all().order_by("id")
    serializer_class = DepartmentSerializer
    # permission_classes = [IsAdminUser]

    def get_permissions(self):
        if self.request.method == "GET":
            return []
        return [IsAdminUser()]


class SingleDepartmentView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Department.objects.all()
    serializer_class = DepartmentSerializer

    def get_permissions(self):
        if self.request.method == "GET":
            return []
        return [IsAdminUser()]


class CompanyView(generics.ListCreateAPIView):
    queryset = Company.objects.all().order_by("id")
    serializer_class = CompanySerializer

    def get_permissions(self):
        if self.request.method == "GET":
            return []
        return [IsAdminUser()]


class SingleCompanyView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Company.objects.all()
    serializer_class = CompanySerializer

    def get_permissions(self):
        if self.request.method == "GET":
            return []
        return [IsAdminUser()]


class StatusView(generics.ListCreateAPIView):
    queryset = Status.objects.all()
    serializer_class = StatusSerializer
    permission_classes = [IsAdminUser]


class SingleStatusView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Status.objects.all()
    serializer_class = StatusSerializer
    permission_classes = [IsAdminUser]


class ResultsView(generics.ListCreateAPIView):
    queryset = Result.objects.all()
    serializer_class = ResultSerializer
    permission_classes = [IsAdminUser]


class SingleResultView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Result.objects.all()
    serializer_class = ResultSerializer
    permission_classes = [IsAdminUser]


# class RecruiterRequestView(generics.ListCreateAPIView):
#     queryset = RecruiterRequest.objects.all()
#     serializer_class = RecruiterRequestSerializer

#     def get_permissions(self):
#         if self.request.method == "POST":
#             return [IsAuthenticated()]
#         return [IsAdminUser()]


# class SingleRecruiterRequestView(generics.RetrieveUpdateDestroyAPIView):
#     queryset = RecruiterRequest.objects.all()
#     serializer_class = RecruiterRequestSerializer
#     permission_classes = [IsAdminUser]


class JobView(generics.ListCreateAPIView):
    queryset = Job.objects.select_related("department", "company").all().order_by("id")
    serializer_class = JobSerializer

    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    search_fields = ["title"]
    # ordering_fields = ["title", "end_date", "company__name", "department__title"]
    filterset_fields = ["company"]

    def get_permissions(self):
        if self.request.method == "GET":
            return []
        return [isRecruiter()]


class SingleJobView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Job.objects.select_related("department", "company").all()
    serializer_class = JobSerializer

    def get_permissions(self):
        if self.request.method == "GET":
            return []
        return [isRecruiter()]


class ApplicationView(generics.ListCreateAPIView):
    queryset = Application.objects.select_related("user", "job", "status").all()
    serializer_class = ApplicationSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        user = self.request.user
        if user.is_staff:
            queryset = Application.objects.select_related("user", "job", "status").all()
            return queryset
        elif user.groups.filter(name="Recruiter").exists():
            queryset = Application.objects.select_related(
                "user", "job", "status"
            ).filter(job__recruiter=user)
            return queryset
        else:
            queryset = Application.objects.select_related(
                "user", "job", "status"
            ).filter(user=user)
            return queryset

    def perform_create(self, serializer):
        status = Status.objects.get(pk=1)  # Default to 'Pending'
        resume = self.request.FILES.get("resume")
        application = serializer.save(status=status)
        if resume:
            # Save the resume and get the file path
            file_path = default_storage.save(f"resumes/{resume.name}", resume)
            # Remove the full_path line completely
            try:
                screening_service = ResumeScreeningService()
                result = screening_service.screen_resume(
                    file_path, application.job
                )  # Use file_path directly
                # Update application with screening results
                new_status_id = result["status_id"]  # could be 1 or 2 or 3
                new_status = Status.objects.get(pk=new_status_id)
                application.status = new_status
                application.match_score = result["match_score"]
                application.save()
                if new_status_id == 2:
                    default_result = Result.objects.get(pk=3)
                    Interview.objects.create(
                        application=application,
                        date=None,  # Date can remain empty for now
                        result=default_result,
                    )

            except Exception as e:
                print(f"Error in resume screening: {e}")


class SingleApplicationView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Application.objects.select_related("user", "job", "status").all()
    serializer_class = ApplicationSerializer

    def get_permissions(self):
        if self.request.method == "GET":
            return [IsAuthenticated()]
        return [isRecruiter()]

    def get_queryset(self):
        user = self.request.user
        if user.is_staff:
            queryset = Application.objects.select_related("user", "job", "status").all()
            return queryset
        elif user.groups.filter(name="Recruiter").exists():
            queryset = Application.objects.select_related(
                "user", "job", "status"
            ).filter(job__recruiter=user)
            return queryset
        else:
            queryset = Application.objects.select_related(
                "user", "job", "status"
            ).filter(user=user)
            return queryset

    def perform_update(self, serializer):
        user = self.request.user
        application = self.get_object()
        if user.is_staff or application.job.recruiter == user:
            previous_status_id = application.status.id

            updated_application = serializer.save()

            current_status_id = updated_application.status.id
            if current_status_id == 2 and previous_status_id != 2:
                if not Interview.objects.filter(
                    application=updated_application
                ).exists():
                    default_result = Result.objects.get(pk=3)  # Get default result
                    Interview.objects.create(
                        application=updated_application,
                        date=None,  # Date can remain empty for now
                        result=default_result,
                    )

    def perform_destroy(self, instance):
        user = self.request.user
        if user.is_staff or instance.job.recruiter == user:
            instance.delete()


class InterviewView(generics.ListCreateAPIView):
    queryset = Interview.objects.select_related("application", "result").all()
    serializer_class = InterviewSerializer
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def get_permissions(self):
        if self.request.method == "GET":
            return [IsAuthenticated()]
        return [isRecruiter()]

    def get_queryset(self):
        user = self.request.user
        if user.is_staff:
            queryset = Interview.objects.select_related("application", "result").all()
            return queryset
        elif user.groups.filter(name="Recruiter").exists():
            queryset = Interview.objects.select_related("application", "result").filter(
                application__job__recruiter=user
            )
            return queryset
        else:
            queryset = Interview.objects.select_related("application", "result").filter(
                application__user=user
            )
            return queryset

    def perform_create(self, serializer):
        result = Result.objects.get(pk=3)
        interview = serializer.save(result=result)

        # Check if a video file was uploaded
        if "interview_video" in self.request.FILES:
            interview_video = self.request.FILES["interview_video"]
            if interview_video:
                self.process_interview_video(interview, interview_video)

    def process_interview_video(self, interview, video_file):
        try:
            interview.interview_video = video_file
            interview.save()

            # full_path = interview.interview_video.path wont work as files in r2
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                interview.interview_video.open("rb")
                for chunk in interview.interview_video.chunks():
                    temp_file.write(chunk)
                interview.interview_video.close()
                full_path = temp_file.name

            try:
                analysis_service = InterviewAnalysisService()
                analysis_result = analysis_service.process_recording(full_path)

                interview.update_result_from_analysis(analysis_result)

            except Exception as e:
                # Log the error but don't raise it
                print(f"Error in video analysis: {e}")
            finally:
                # Clean up
                try:
                    os.unlink(full_path)
                except OSError:
                    pass

        except Exception as e:
            print(f"Error saving interview video: {e}")


class SingleInterviewView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Interview.objects.select_related("application", "result").all()
    serializer_class = InterviewSerializer
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def get_permissions(self):
        if self.request.method == "GET":
            return [IsAuthenticated()]
        return [isRecruiter()]

    def get_queryset(self):
        user = self.request.user
        if user.is_staff:
            queryset = Interview.objects.select_related("application", "result").all()
            return queryset
        elif user.groups.filter(name="Recruiter").exists():
            queryset = Interview.objects.select_related("application", "result").filter(
                application__job__recruiter=user
            )
            return queryset
        else:
            queryset = Interview.objects.select_related("application", "result").filter(
                application__user=user
            )
            return queryset

    def perform_update(self, serializer):
        user = self.request.user
        interview = self.get_object()

        if user.is_staff or interview.application.job.recruiter == user:
            # Save the updated interview first
            updated_interview = serializer.save()

            # Then check if a new video was uploaded
            if "interview_video" in self.request.FILES:
                interview_video = self.request.FILES["interview_video"]
                if interview_video:
                    self.process_interview_video(updated_interview, interview_video)

    def process_interview_video(self, interview, video_file):
        try:
            # Save the video file directly to the model
            interview.interview_video = video_file
            interview.save()

            # Get the full path to the saved file
            # full_path = interview.interview_video.path cant do this for cloud storage r2 stored files
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                interview.interview_video.open("rb")
                for chunk in interview.interview_video.chunks():
                    temp_file.write(chunk)
                interview.interview_video.close()
                full_path = temp_file.name

            # Now process the video for analysis
            try:
                # Analyze the video using the interview analysis service
                analysis_service = InterviewAnalysisService()
                analysis_result = analysis_service.process_recording(full_path)

                # Update the interview with the analysis results
                interview.update_result_from_analysis(analysis_result)

            except Exception as e:
                # Log the error but don't raise it
                print(f"Error in video analysis: {e}")

            finally:
                # Clean up
                try:
                    os.unlink(full_path)
                except OSError:
                    pass

        except Exception as e:
            print(f"Error saving interview video: {e}")

    def perform_destroy(self, instance):
        user = self.request.user
        if user.is_staff or instance.application.job.recruiter == user:
            instance.delete()


class PredictedCandidateView(generics.ListCreateAPIView):
    queryset = PredictedCandidate.objects.select_related("interview", "status").all()
    serializer_class = PredictedCandidateSerializer

    def get_permissions(self):
        if self.request.method == "GET":
            return [IsAuthenticated()]
        return [isRecruiter()]

    def get_queryset(self):
        user = self.request.user
        if user.is_staff:
            queryset = PredictedCandidate.objects.select_related(
                "interview", "status"
            ).all()
            return queryset
        elif user.groups.filter(name="Recruiter").exists():
            queryset = PredictedCandidate.objects.select_related(
                "interview", "status"
            ).filter(interview__application__job__recruiter=user)
            return queryset
        else:
            queryset = PredictedCandidate.objects.select_related(
                "interview", "status"
            ).filter(interview__application__user=user)
            return queryset


class SinglePredictedCandidateView(generics.RetrieveUpdateDestroyAPIView):
    queryset = PredictedCandidate.objects.select_related("interview", "status").all()
    serializer_class = PredictedCandidateSerializer

    def get_permissions(self):
        if self.request.method == "GET":
            return [IsAuthenticated()]
        return [isRecruiter()]

    def get_queryset(self):
        user = self.request.user
        if user.is_staff:
            queryset = PredictedCandidate.objects.select_related(
                "interview", "status"
            ).all()
            return queryset
        elif user.groups.filter(name="Recruiter").exists():
            queryset = PredictedCandidate.objects.select_related(
                "interview", "status"
            ).filter(interview__application__job__recruiter=user)
            return queryset
        else:
            queryset = PredictedCandidate.objects.select_related(
                "interview", "status"
            ).filter(interview__application__user=user)
            return queryset


class EvaluationFormView(APIView):
    """API endpoint to handle candidate evaluation form submissions"""

    permission_classes = [isRecruiter]

    def post(self, request, pk=None):
        try:
            # Get the predicted candidate
            candidate = get_object_or_404(PredictedCandidate, pk=pk)

            user = request.user
            if not (
                user.is_staff or candidate.interview.application.job.recruiter == user
            ):
                return Response(
                    {"error": "You do not have permission to evaluate this candidate."},
                    status=status.HTTP_403_FORBIDDEN,
                )

            evaluation_data = request.data.get("evaluation_data")
            if not evaluation_data:
                return Response(
                    {"error": "No evaluation data provided."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Calculate average score
            questions = evaluation_data.get("questions", [])
            if not questions:
                return Response(
                    {"error": "No questions data provided."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            total_score = 0
            question_count = len(questions)

            for question in questions:
                score = question.get("score", 0)
                total_score += score

            # Calculate average
            average_score = total_score / question_count if question_count > 0 else 0

            status_id = 2 if average_score >= 3.5 else 3  # 2 = Hired, 3 = Rejected

            candidate.evaluation_data = evaluation_data
            candidate.evaluation_score = average_score
            candidate.status_id = status_id
            candidate.save()

            service = CandidateEvaluationService()
            updated_candidate = service.evaluate_candidate(candidate, evaluation_data)
            return Response(
                {
                    "success": True,
                    "message": "Evaluation submitted successfully",
                    "status": updated_candidate.status.title,
                    "average_score": updated_candidate.evaluation_score,
                }
            )

        except Exception as e:
            return Response(
                {"error": f"Failed to submit evaluation: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


# class Recruiter(APIView):
#     permission_classes = [IsAdminUser]

#     def get(self, request):
#         group = get_object_or_404(Group, name="Recruiter")
#         users = group.user_set.all()
#         users_data = [{"username": user.username, "id": user.pk} for user in users]
#         return Response(users_data, status=status.HTTP_200_OK)

#     def post(self, request):
#         username = request.data.get("username")
#         if username:
#             user = get_object_or_404(User, username=username)
#             group = get_object_or_404(Group, name="Recruiter")
#             group.user_set.add(user)
#             return Response(
#                 {"message": f"User {username} promoted to recruiter."},
#                 status=status.HTTP_200_OK,
#             )


# class SingleRecruiter(APIView):
#     permission_classes = [IsAdminUser]

#     def delete(self, request, userID):
#         user = get_object_or_404(User, id=userID)
#         group = get_object_or_404(Group, name="Recruiter")
#         group.user_set.remove(user)
#         return Response(
#             {"message": "User removed from Recruiter"}, status=status.HTTP_200_OK
#         )


class CustomTokenObtainPairView(TokenObtainPairView):
    serializer_class = CustomTokenObtainPairSerializer

    def post(self, request, *args, **kwargs):
        response = super().post(request, *args, **kwargs)
        if response.status_code == 200:
            token = response.data
            response_data = {"access": token["access"], "role": token["role"]}
            new_response = Response(response_data, status=status.HTTP_200_OK)

            cookie_max_age = int(
                settings.SIMPLE_JWT["REFRESH_TOKEN_LIFETIME"].total_seconds()
            )

            new_response.set_cookie(
                "refresh_token",
                token["refresh"],
                max_age=cookie_max_age,
                httponly=True,
                secure=True,
                samesite="None",
                domain=None,
            )

            return new_response
        else:
            print(f"LOGIN: Failed with status {response.status_code}: {response.data}")
        return response


class CustomTokenRefreshView(TokenRefreshView):
    serializer_class = CustomTokenRefreshSerializer

    def post(self, request, *args, **kwargs):
        # Get refresh token from HttpOnly cookie
        refresh_token = request.COOKIES.get("refresh_token")
        if not refresh_token:
            return Response(
                {"detail": "Refresh token not found in cookies"},
                status=status.HTTP_401_UNAUTHORIZED,
            )
        # Add refresh token to request data
        request.data["refresh"] = refresh_token
        response = super().post(request, *args, **kwargs)
        if response.status_code == 200:
            token_data = response.data
            # If rotation is enabled, we get a new refresh token
            if "refresh" in token_data:
                # Update the refresh token cookie
                response.set_cookie(
                    "refresh_token",
                    token_data["refresh"],
                    max_age=settings.SIMPLE_JWT[
                        "REFRESH_TOKEN_LIFETIME"
                    ].total_seconds(),
                    httponly=True,
                    secure=True,
                    samesite="None",
                    domain=None,
                )
                # Remove refresh token from response body
                del token_data["refresh"]
        else:
            print(
                f"REFRESH: Failed with status {response.status_code}: {response.data}"
            )

        return response


@api_view(["POST"])
@permission_classes([AllowAny])
def logout_view(request):
    """Clear the refresh token cookie"""
    response = Response(
        {"detail": "Successfully logged out"}, status=status.HTTP_200_OK
    )
    response.delete_cookie("refresh_token")
    return response
