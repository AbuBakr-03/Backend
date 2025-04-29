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


# APIBackend/models.py - Update the Application model
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


# APIBackend/models.py - Update the Interview model
class Interview(models.Model):
    application = models.ForeignKey(Application, on_delete=models.CASCADE, null=None)
    date = models.DateTimeField(null=True, blank=True)
    result = models.ForeignKey(Result, on_delete=models.CASCADE, null=None, default=1)
    external_meeting_link = models.CharField(max_length=255, null=True, blank=True)
    interview_video = models.FileField(upload_to="interviews/", null=True, blank=True)
    analysis_data = models.JSONField(null=True, blank=True)
    interview_questions = models.JSONField(null=True, blank=True)  # Add this field

    class Meta:
        unique_together = ("application",)

    # Add this method to your Interview model in models.py if it's missing

    def update_result_from_analysis(self, analysis_data):

        self.analysis_data = analysis_data

        # Update the result if confidence meets thresholds
        confidence = analysis_data.get("confidence", 0)  # 0 if no data
        if confidence >= 50:
            self.result_id = 2  # Assuming 2 is the ID for "Hired/Accepted"
        else:
            self.result_id = 3  # Assuming 3 is the ID for "Rejected"

        self.save()

    def __str__(self):
        return f"{self.application} : {self.date} : {self.result}"


class RecruiterRequest(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=None)
