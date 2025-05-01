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
    EvaluationStatus,
    PredictedCandidate,
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


# APIBackend/serializers.py - Update ApplicationSerializer
class ApplicationSerializer(serializers.ModelSerializer):
    status = StatusSerializer(read_only=True)
    job = JobSerializer(read_only=True)
    job_id = serializers.IntegerField(write_only=True)
    user = serializers.HiddenField(default=serializers.CurrentUserDefault())
    resume = serializers.FileField(required=False)
    match_score = serializers.FloatField(read_only=True)

    class Meta:
        model = Application
        fields = [
            "id",
            "name",
            "email",
            "residence",
            "cover_letter",
            "resume",
            "user",
            "job",
            "job_id",
            "status",
            "match_score",
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
    external_meeting_link = serializers.CharField(required=False, allow_null=True)
    interview_video = serializers.FileField(required=False, allow_null=True)
    analysis_data = serializers.JSONField(read_only=True)
    interview_questions = serializers.JSONField(read_only=True)

    class Meta:
        model = Interview
        fields = [
            "id",
            "application",
            "application_id",
            "date",
            "result",
            "external_meeting_link",
            "interview_video",
            "analysis_data",
            "interview_questions",
        ]


class EvaluationStatusSerializer(serializers.ModelSerializer):
    class Meta:
        model = EvaluationStatus
        fields = "__all__"

class PredictedCandidateSerializer(serializers.ModelSerializer):
    interview = InterviewSerializer(read_only=True)
    status = EvaluationStatusSerializer(read_only=True)

    class Meta:
        model = PredictedCandidate
        fields = ["id", "interview", "status", "evaluation_score", "evaluation_data"]


from rest_framework_simplejwt.serializers import TokenObtainPairSerializer


class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)

        token["is_staff"] = user.is_staff
        token["is_superuser"] = user.is_superuser
        token["is_recruiter"] = user.groups.filter(name="Recruiter").exists()

        return token

    def validate(self, attrs):
        data = super().validate(attrs)

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
