from django.contrib.auth.models import User
from rest_framework import serializers
from .models import (
    Department,
    Job,
    Interview,
    Result,
    Company,
    Status,
    Application,
    RecruiterRequest,
)


class RecruiterRequestSerializer(serializers.ModelSerializer):
    user = serializers.HiddenField(default=serializers.CurrentUserDefault())
    username = serializers.CharField(source="user.username", read_only=True)

    class Meta:
        model = RecruiterRequest
        fields = ["id", "user", "username"]


class DepartmentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Department
        fields = "__all__"


class CompanySerializer(serializers.ModelSerializer):
    class Meta:
        model = Company
        fields = "__all__"


class JobSerializer(serializers.ModelSerializer):
    department = DepartmentSerializer(read_only=True)
    department_id = serializers.IntegerField(write_only=True)
    company = CompanySerializer(read_only=True)
    company_id = serializers.IntegerField(write_only=True)
    recruiter = serializers.HiddenField(default=serializers.CurrentUserDefault())

    class Meta:
        model = Job
        fields = [
            "id",
            "title",
            "location",
            "responsiblities",
            "qualification",
            "nice_to_haves",
            "end_date",
            "department",
            "department_id",
            "company",
            "company_id",
            "recruiter",
        ]


class StatusSerializer(serializers.ModelSerializer):
    class Meta:
        model = Status
        fields = "__all__"


class ApplicationSerializer(serializers.ModelSerializer):
    status = StatusSerializer(read_only=True)
    job = JobSerializer(read_only=True)
    job_id = serializers.IntegerField(write_only=True)
    user = serializers.HiddenField(default=serializers.CurrentUserDefault())

    class Meta:
        model = Application
        fields = [
            "id",
            "name",
            "email",
            "residence",
            "cover_letter",
            "user",
            "job",
            "job_id",
            "status",
        ]


class ResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = Result
        fields = "__all__"


class InterviewSerializer(serializers.ModelSerializer):
    application = ApplicationSerializer(read_only=True)
    application_id = serializers.IntegerField(write_only=True)
    result = ResultSerializer(read_only=True)
    date = serializers.DateTimeField(required=False, allow_null=True)

    class Meta:
        model = Interview
        fields = ["id", "application", "application_id", "date", "result"]

    def validate(self, attrs):
        application_id = attrs.get("application_id")
        date = attrs.get("date")

        # Only validate if date is provided
        if date is not None and application_id is not None:
            try:
                application = Application.objects.select_related("job__recruiter").get(
                    id=application_id
                )
            except Application.DoesNotExist:
                raise serializers.ValidationError("Invalid application ID.")

            recruiter = application.job.recruiter
            if recruiter is None:
                raise serializers.ValidationError("This job has no assigned recruiter.")

            # Look for conflicting interviews with same recruiter and date
            existing = Interview.objects.filter(
                application__job__recruiter=recruiter, date=date
            )

            # Exclude the current instance if this is an update
            if self.instance:
                existing = existing.exclude(pk=self.instance.pk)

            if existing.exists():
                raise serializers.ValidationError(
                    f"The recruiter already has an interview scheduled at {date}."
                )

        return attrs


# APIBackend/serializers.py

from rest_framework_simplejwt.serializers import TokenObtainPairSerializer


class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)

        # Add custom claims to the token itself
        token["is_staff"] = user.is_staff
        token["is_superuser"] = user.is_superuser
        token["is_recruiter"] = user.groups.filter(name="Recruiter").exists()

        return token

    def validate(self, attrs):
        data = super().validate(attrs)

        # Add extra user info outside of the token
        data.update(
            {
                "user": {
                    "id": self.user.id,
                    "email": self.user.email,
                    "is_staff": self.user.is_staff,
                    "is_superuser": self.user.is_superuser,
                    "is_recruiter": self.user.groups.filter(name="Recruiter").exists(),
                }
            }
        )

        return data
