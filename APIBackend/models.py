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
        return f"{self.title} : {self.location} : {self.end_date} : {self.department} : {self.company} : {self.recruiter}"


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
        return f"{self.name} : {self.email} : {self.residence} : {self.user} : {self.job} : {self.status}"


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
    meeting_link = models.CharField(max_length=255, null=True, blank=True)
    meeting_id = models.CharField(
        max_length=50, null=True, blank=True
    )  # For unique meeting identification

    def __str__(self):
        return f"{self.application} : {self.date} : {self.result}"

    def generate_meeting_link(self):
        """Generate a unique meeting link if one doesn't exist"""
        if not self.meeting_id:
            # Generate a unique ID for the meeting
            import uuid

            self.meeting_id = str(uuid.uuid4())[:8]
            self.meeting_link = f"http://127.0.0.1:5180/meeting/{self.meeting_id}"
            self.save()
        return self.meeting_link


class RecruiterRequest(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=None)
