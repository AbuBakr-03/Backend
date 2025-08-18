# Update to APIBackend/urls.py
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views
from . import views_interview
from . import views_questions
# 
urlpatterns = [
    path("department/", views.DepartmentView.as_view()),
    path("department/<int:pk>/", views.SingleDepartmentView.as_view()),
    path("company/", views.CompanyView.as_view()),
    path("company/<int:pk>/", views.SingleCompanyView.as_view()),
    path("job/", views.JobView.as_view()),
    path("job/<int:pk>/", views.SingleJobView.as_view()),
    path("status/", views.StatusView.as_view()),
    path("status/<int:pk>/", views.SingleStatusView.as_view()),
    path("application/", views.ApplicationView.as_view()),
    path("application/<int:pk>/", views.SingleApplicationView.as_view()),
    path("result/", views.ResultsView.as_view()),
    path("result/<int:pk>/", views.SingleResultView.as_view()),
    path("interview/", views.InterviewView.as_view()),
    path("interview/<int:pk>/", views.SingleInterviewView.as_view()),
    # New endpoint for interview recording analysis
    path(
        "interview/<int:pk>/analyze-recording/",
        views_interview.InterviewRecordingView.as_view(),
        name="analyze-recording",
    ),
    # path("recruiters/", views.RecruiterRequestView.as_view()),
    # path("recruiters/<int:pk>/", views.SingleRecruiterRequestView.as_view()),
    # path("groups/recruiters/", views.Recruiter.as_view()),
    # path("groups/recruiters/<int:pk>/", views.SingleRecruiter.as_view()),
    path(
        "interview/<int:pk>/generate-questions/",
        views_questions.InterviewQuestionsView.as_view(),
        name="generate-questions",
    ),
    # Add to APIBackend/urls.py
    # PredictedCandidate URLs
    path("predicted-candidates/", views.PredictedCandidateView.as_view()),
    path(
        "predicted-candidates/<int:pk>/", views.SinglePredictedCandidateView.as_view()
    ),
    path(
        "predicted-candidates/<int:pk>/evaluate/",
        views.EvaluationFormView.as_view(),
    ),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
