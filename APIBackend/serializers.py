from djoser.serializers import UserCreateSerializer
from django.contrib.auth.models import User, Group
from rest_framework import serializers
from .models import (
    Department,
    Job,
    Interview,
    Result,
    Company,
    Status,
    Application,
    # RecruiterRequest,
    EvaluationStatus,
    PredictedCandidate,
)


# class RecruiterRequestSerializer(serializers.ModelSerializer):
#     user = serializers.HiddenField(default=serializers.CurrentUserDefault())
#     username = serializers.CharField(source="user.username", read_only=True)

#     class Meta:
#         model = RecruiterRequest
#         fields = ["id", "user", "username"]


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


from rest_framework_simplejwt.serializers import (
    TokenObtainPairSerializer,
    TokenRefreshSerializer,
)
from rest_framework_simplejwt.tokens import RefreshToken


class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    def validate(self, attrs):
        data = super().validate(attrs)
        data["role"] = (
            "admin"
            if self.user.is_superuser
            else (
                "Recruiter"
                if self.user.groups.filter(name="Recruiter").exists()
                else "user"
            )
        )
        return data


class CustomUserCreateSerializer(UserCreateSerializer):
    group = serializers.CharField(write_only=True, required=False, allow_blank=True)

    class Meta(UserCreateSerializer.Meta):
        fields = UserCreateSerializer.Meta.fields + ("group",)

    def validate(self, attrs):
        # Remove group from attrs before parent validation
        group_name = attrs.pop("group", None)
        # Call parent validation
        validated_data = super().validate(attrs)
        # Add group back to validated data for use in create method
        if group_name:
            validated_data["group"] = group_name
        return validated_data

    def create(self, validated_data):
        group_name = validated_data.get("group")
        validated_data.pop("group", None)
        user = super().create(validated_data)

        if group_name:
            group = Group.objects.get(name=group_name)
            user.groups.add(group)

        return user


class CustomTokenRefreshSerializer(TokenRefreshSerializer):
    def validate(self, attrs):
        # Get refresh token from cookie if not in attrs
        if "refresh" not in attrs:
            request = self.context.get("request")
            if request:
                refresh_token = request.COOKIES.get("refresh_token")
                if refresh_token:
                    attrs["refresh"] = refresh_token
                else:
                    raise serializers.ValidationError("No refresh token found")
            else:
                raise serializers.ValidationError("No request context available")
        
        data = super().validate(attrs)
        # Get the refresh token and extract user info
        refresh = RefreshToken(attrs["refresh"])
        user_id = refresh.payload.get("user_id")

        if user_id:
            from django.contrib.auth import get_user_model

            User = get_user_model()
            try:
                user = User.objects.get(id=user_id)
                data["role"] = (
                    "admin"
                    if user.is_superuser
                    else (
                        "Recruiter"
                        if user.groups.filter(name="Recruiter").exists()
                        else "user"
                    )
                )

            except User.DoesNotExist:
                data["role"] = "user"
        else:
            data["role"] = "user"

        return data
