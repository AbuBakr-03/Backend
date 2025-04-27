from django.contrib import admin
from django.urls import path, include
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)
from APIBackend.views import CustomTokenObtainPairView
from rest_framework_simplejwt.views import TokenBlacklistView

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("APIBackend.urls")),
    path("auth/", include("djoser.urls")),
    path("auth/jwt/create/", CustomTokenObtainPairView.as_view(), name="jwt-create"),
    path("auth/jwt/refresh/", TokenRefreshView.as_view(), name="jwt-refresh"),
    path(
        "auth/jwt/logout/", TokenBlacklistView.as_view(), name="jwt-logout"
    ),  # <-- add this
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
