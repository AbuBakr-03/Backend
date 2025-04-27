from django.contrib.auth.models import User, Group
from django.shortcuts import get_object_or_404
from rest_framework import generics, status
from rest_framework.views import Response
from rest_framework.decorators import permission_classes, APIView
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.filters import SearchFilter, OrderingFilter
from rest_framework.permissions import IsAuthenticated, IsAdminUser, BasePermission
from .serializers import (
    DepartmentSerializer,
    CompanySerializer,
    JobSerializer,
    StatusSerializer,
    ApplicationSerializer,
    ResultSerializer,
    InterviewSerializer,
    RecruiterRequestSerializer,
)
from .models import (
    Department,
    Company,
    Job,
    Status,
    Application,
    Result,
    Interview,
    RecruiterRequest,
)


from rest_framework_simplejwt.views import TokenObtainPairView
from .serializers import CustomTokenObtainPairSerializer


from .services import ResumeScreeningService
from django.core.files.storage import default_storage
import os
from BackendProject import settings

from rest_framework.decorators import action
from rest_framework.response import Response


class CustomTokenObtainPairView(TokenObtainPairView):
    serializer_class = CustomTokenObtainPairSerializer


# Create your views here.
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
            return [isRecruiter()]
        return [IsAdminUser()]


class SingleDepartmentView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Department.objects.all()
    serializer_class = DepartmentSerializer
    permission_classes = [IsAdminUser]


class CompanyView(generics.ListCreateAPIView):
    queryset = Company.objects.all().order_by("id")
    serializer_class = CompanySerializer

    def get_permissions(self):
        if self.request.method == "GET":
            return [isRecruiter()]
        return [IsAdminUser()]


class SingleCompanyView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Company.objects.all()
    serializer_class = CompanySerializer
    permission_classes = [IsAdminUser]


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


class RecruiterRequestView(generics.ListCreateAPIView):
    queryset = RecruiterRequest.objects.all()
    serializer_class = RecruiterRequestSerializer

    def get_permissions(self):
        if self.request.method == "POST":
            return []
        return [IsAdminUser()]


class SingleRecruiterRequestView(generics.RetrieveUpdateDestroyAPIView):
    queryset = RecruiterRequest.objects.all()
    serializer_class = RecruiterRequestSerializer
    permission_classes = [IsAdminUser]


class JobView(generics.ListCreateAPIView):
    queryset = Job.objects.select_related("department", "company").all().order_by("id")
    serializer_class = JobSerializer
    filter_backends = [DjangoFilterBackend, OrderingFilter, SearchFilter]
    filterset_fields = ["location", "department__title", "company__name"]
    search_fields = ["title"]
    ordering_fields = ["end_date"]

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

        # Save application with default status first
        application = serializer.save(status=status)

        # If resume is uploaded, process it
        if resume:
            # Save the resume and get the file path
            file_path = default_storage.save(f"resumes/{resume.name}", resume)
            full_path = os.path.join(settings.MEDIA_ROOT, file_path)

            try:
                # Use the screening service to evaluate the resume
                screening_service = ResumeScreeningService()
                result = screening_service.screen_resume(full_path, application.job)

                # Update application with screening results
                new_status_id = result["status_id"]
                new_status = Status.objects.get(pk=new_status_id)
                application.status = new_status
                application.match_score = result["match_score"]
                application.save()

                if new_status_id == 2:
                    default_result = Result.objects.get(
                        pk=1
                    )  # Get default result (usually 'Pending')
                    Interview.objects.create(
                        application=application,
                        date=None,  # Date can remain empty for now
                        result=default_result,
                    )

            except Exception as e:
                print(f"Error in resume screening: {e}")
                # Keep the default status on error


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
            # Get previous status
            previous_status_id = application.status.id

            # Save the updated application
            updated_application = serializer.save()

            # If status changed to 'Approved for Interview' (ID 2), create an interview
            current_status_id = updated_application.status.id
            if current_status_id == 2 and previous_status_id != 2:
                # Check if an interview already exists
                if not Interview.objects.filter(
                    application=updated_application
                ).exists():
                    default_result = Result.objects.get(pk=1)  # Get default result
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
        result = Result.objects.get(pk=1)
        serializer.save(result=result)


class SingleInterviewView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Interview.objects.select_related("application", "result").all()
    serializer_class = InterviewSerializer

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
            serializer.save()

    def perform_destroy(self, instance):
        user = self.request.user
        if user.is_staff or instance.application.job.recruiter == user:
            instance.delete()


# Remove the @action methods from both InterviewView and SingleInterviewView


# Add this new view
class GenerateMeetingLinkView(APIView):
    permission_classes = [isRecruiter]

    def post(self, request, pk=None):
        try:
            interview = Interview.objects.get(pk=pk)
            # Check permission
            user = request.user
            if not (user.is_staff or interview.application.job.recruiter == user):
                return Response(
                    {"error": "You do not have permission to modify this interview."},
                    status=status.HTTP_403_FORBIDDEN,
                )

            # Generate meeting link
            meeting_link = interview.generate_meeting_link()
            return Response({"meeting_link": meeting_link})
        except Interview.DoesNotExist:
            return Response(
                {"error": "Interview not found."}, status=status.HTTP_404_NOT_FOUND
            )


class Recruiter(APIView):
    permission_classes = [IsAdminUser]

    def get(self, request):
        group = get_object_or_404(Group, name="Recruiter")
        users = group.user_set.all()
        users_data = [{"username": user.username, "id": user.pk} for user in users]
        return Response(users_data, status=status.HTTP_200_OK)

    def post(self, request):
        username = request.data.get("username")
        if username:
            user = get_object_or_404(User, username=username)
            group = get_object_or_404(Group, name="Recruiter")
            group.user_set.add(user)
            return Response(
                {"message": f"User {username} promoted to recruiter."},
                status=status.HTTP_200_OK,
            )
            return Response(
                {"error": "Username not provided."}, status=status.HTTP_400_BAD_REQUEST
            )


class SingleRecruiter(APIView):
    permission_classes = [IsAdminUser]

    def delete(self, request, userID):
        user = get_object_or_404(User, id=userID)
        group = get_object_or_404(Group, name="Recruiter")
        group.user_set.remove(user)
        return Response(
            {"message": "User removed from Recruiter"}, status=status.HTTP_200_OK
        )
