from django.contrib import admin
from .models import (
    Department,
    Company,
    Job,
    Status,
    Application,
    Result,
    Interview,
    # RecruiterRequest,
    EvaluationStatus,
    PredictedCandidate,
)

# Register your models here.
admin.site.register(Department)
admin.site.register(Company)
admin.site.register(Job)
admin.site.register(Status)
admin.site.register(Application)
admin.site.register(Result)
admin.site.register(Interview)
# admin.site.register(RecruiterRequest)
admin.site.register(EvaluationStatus)
admin.site.register(PredictedCandidate)
