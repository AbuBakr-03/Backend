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
        status = Status.objects.get(pk=1)
        serializer.save(status=status)


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
            serializer.save()

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
