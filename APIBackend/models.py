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


class Application(models.Model):
    name = models.CharField(max_length=255)
    email = models.EmailField()
    residence = models.CharField(max_length=255)
    cover_letter = models.TextField(max_length=1000)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=None)
    job = models.ForeignKey(Job, on_delete=models.CASCADE, null=None)
    status = models.ForeignKey(Status, on_delete=models.CASCADE, null=None, default=1)

    class Meta:
        unique_together = ["user", "job"]

    def __str__(self):
        return f"{self.name} : {self.email} : {self.residence} : {self.user} : {self.job} : {self.status}"


class Result(models.Model):
    title = models.CharField(max_length=255)
    slug = models.SlugField()

    def __str__(self):
        return self.title


class Interview(models.Model):
    application = models.ForeignKey(Application, on_delete=models.CASCADE, null=None)
    date = models.DateTimeField(null=True, blank=True)
    result = models.ForeignKey(Result, on_delete=models.CASCADE, null=None, default=1)

    def __str__(self):
        return f"{self.application} : {self.date} : {self.result}"


class RecruiterRequest(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=None)
