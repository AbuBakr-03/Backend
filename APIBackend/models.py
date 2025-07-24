from django.db import models
from django.contrib.auth.models import User


# Create your models here.
class Department(models.Model):
    title = models.CharField(max_length=255)
    slug = models.SlugField()

    def __str__(self):
        return self.title


class Company(models.Model):
    name = models.CharField(max_length=255)
    slug = models.SlugField()
    logo = models.ImageField(upload_to="logos/", default="logos/building-2.png")

    def __str__(self):
        return self.name


class Job(models.Model):
    title = models.CharField(max_length=255)
    location = models.CharField(max_length=255)
    responsiblities = models.TextField(max_length=1000)
    qualification = models.TextField(max_length=1000)
    nice_to_haves = models.TextField(max_length=1000)
    end_date = models.DateTimeField()
    department = models.ForeignKey(Department, on_delete=models.CASCADE, null=None)
    company = models.ForeignKey(Company, on_delete=models.CASCADE, null=None)
    recruiter = models.ForeignKey(
        User, related_name="recruiter", on_delete=models.SET_NULL, null=True
    )

    class Meta:
        unique_together = ["company", "department"]

    def __str__(self):
        return f"{self.title} : {self.end_date} : {self.department} : {self.company} : {self.recruiter}"


class Status(models.Model):
    title = models.CharField(max_length=255)
    slug = models.SlugField()

    def __str__(self):
        return self.title


class Application(models.Model):
    name = models.CharField(max_length=255)
    email = models.EmailField()
    residence = models.CharField(max_length=255)
    cover_letter = models.TextField(max_length=1000)
    resume = models.FileField(upload_to="resumes/", null=True, blank=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=None)
    job = models.ForeignKey(Job, on_delete=models.CASCADE, null=None)
    status = models.ForeignKey(Status, on_delete=models.CASCADE, null=None, default=1)
    match_score = models.FloatField(null=True, blank=True)

    class Meta:
        unique_together = ["user", "job"]

    def __str__(self):
        return f"{self.name} : {self.user} : {self.job} : {self.status} : {self.match_score}"


class Result(models.Model):
    title = models.CharField(max_length=255)
    slug = models.SlugField()

    def __str__(self):
        return self.title


class Interview(models.Model):
    application = models.ForeignKey(Application, on_delete=models.CASCADE, null=None)
    date = models.DateTimeField(null=True, blank=True)
    result = models.ForeignKey(Result, on_delete=models.CASCADE, null=None, default=1)
    external_meeting_link = models.CharField(max_length=255, null=True, blank=True)
    interview_video = models.FileField(upload_to="interviews/", null=True, blank=True)
    analysis_data = models.JSONField(null=True, blank=True)
    interview_questions = models.JSONField(null=True, blank=True)

    class Meta:
        unique_together = ("application",)

    def update_result_from_analysis(self, analysis_data):
        self.analysis_data = analysis_data

        # Update the result if confidence meets thresholds
        confidence = analysis_data.get("confidence", 0)  # 0 if no data
        old_result_id = self.result_id

        if confidence >= 39:
            self.result_id = 2
        else:
            self.result_id = 3

        self.save()

        # Create PredictedCandidate if result changed to "pass"
        if self.result_id == 2 and old_result_id != 2:
            from .models import PredictedCandidate, EvaluationStatus

            # Check if a PredictedCandidate already exists
            if not hasattr(self, "predicted_candidate"):
                # Create with default status (1 = Pending)
                default_status = EvaluationStatus.objects.get(id=1)
                PredictedCandidate.objects.create(interview=self, status=default_status)


class EvaluationStatus(models.Model):
    title = models.CharField(max_length=255)
    slug = models.SlugField()

    def __str__(self):
        return self.title


class PredictedCandidate(models.Model):
    interview = models.OneToOneField(
        Interview, on_delete=models.CASCADE, related_name="predicted_candidate"
    )
    status = models.ForeignKey(EvaluationStatus, on_delete=models.CASCADE, default=1)
    evaluation_score = models.FloatField(null=True, blank=True)
    evaluation_data = models.JSONField(null=True, blank=True)

    def __str__(self):
        return f"{self.interview.application.name} - {self.status}"


class RecruiterRequest(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=None)
