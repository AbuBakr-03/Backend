import os
from dotenv import load_dotenv
from urllib.parse import urlparse
from pathlib import Path
from datetime import timedelta
from decouple import config

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = config("SECRET_KEY", cast=str, default="your-fallback-secret-key-here")
GEMINI_API_KEY = config("GEMINI_API_KEY", cast=str, default="")
DEBUG = config("DEBUG", cast=bool, default=False)

# Security settings for production
if not DEBUG:
    SECURE_SSL_REDIRECT = True
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
    SECURE_HSTS_SECONDS = 31536000
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
    SECURE_BROWSER_XSS_FILTER = True
    X_FRAME_OPTIONS = "DENY"

# Railway-specific hostname handling
RAILWAY_EXTERNAL_HOSTNAME = config("RAILWAY_PUBLIC_DOMAIN", default="")
ALLOWED_HOSTS = [
    "localhost",
    "127.0.0.1",
    "*.railway.app",
    RAILWAY_EXTERNAL_HOSTNAME,
]

# Filter out empty hosts
ALLOWED_HOSTS = [host for host in ALLOWED_HOSTS if host]

MEDIA_URL = "/media/"
MEDIA_ROOT = os.path.join(BASE_DIR, "media")

# AI Models storage configuration
IS_RAILWAY = os.getenv("RAILWAY_ENVIRONMENT") is not None
IS_LOCAL_DEV = DEBUG and not IS_RAILWAY

if IS_RAILWAY:
    # On Railway: Store models in /tmp folder
    MODELS_ROOT = "/tmp/ai_models"
    print("🚀 Running on Railway - models will be stored in /tmp/ai_models")
elif IS_LOCAL_DEV:
    # Local development: Use existing AImodels folder
    MODELS_ROOT = os.path.join(BASE_DIR, "APIBackend", "AImodels")
    print("💻 Running locally - using existing AImodels folder")
else:
    # Other production environments
    MODELS_ROOT = os.path.join(BASE_DIR, "models")
    print("🌐 Running in production - models stored in project/models folder")

# Ensure models directory exists
os.makedirs(MODELS_ROOT, exist_ok=True)

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "APIBackend",
    "rest_framework",
    "rest_framework.authtoken",
    "rest_framework_simplejwt.token_blacklist",
    "djoser",
    "django_filters",
    "corsheaders",
    "django_extensions",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",  # For static files
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "BackendProject.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "BackendProject.wsgi.application"

# Database configuration for Railway
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL:
    tmpPostgres = urlparse(DATABASE_URL)
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": tmpPostgres.path.replace("/", ""),
            "USER": tmpPostgres.username,
            "PASSWORD": tmpPostgres.password,
            "HOST": tmpPostgres.hostname,
            "PORT": 5432,
        }
    }
else:
    # Fallback for local development
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# Static files configuration
STATIC_URL = "/static/"
STATIC_ROOT = os.path.join(BASE_DIR, "staticfiles")
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": (
        "rest_framework_simplejwt.authentication.JWTAuthentication",
        "rest_framework.authentication.SessionAuthentication",
    ),
    "DEFAULT_FILTER_BACKENDS": [
        "django_filters.rest_framework.DjangoFilterBackend",
        "rest_framework.filters.SearchFilter",
        "rest_framework.filters.OrderingFilter",
    ],
}

DJOSER = {
    "LOGIN_FIELD": "username",
    "USER_ID_FIELD": "username",
    "PASSWORD_RESET_CONFIRM_URL": "reset-password/{uid}/{token}",
    "SEND_PASSWORD_RESET_EMAIL": True,
    "EMAIL": {
        "password_reset": "APIBackend.email.CustomPasswordResetEmail",
    },
    "SERIALIZERS": {
        "current_user": "djoser.serializers.UserSerializer",
        "create_user": "APIBackend.serializers.CustomUserCreateSerializer",
    },
    "TOKEN_MODEL": None,
    "DOMAIN": config("FRONTEND_URL", default="127.0.0.1:5186"),
    "SITE_NAME": "SmartHR",
}

# Email configuration
EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"
EMAIL_HOST = "smtp.resend.com"
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = "resend"
EMAIL_HOST_PASSWORD = config("RESEND_API_KEY", default="")
DEFAULT_FROM_EMAIL = "noreply@smarthr.website"

SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(minutes=5),
    "REFRESH_TOKEN_LIFETIME": timedelta(minutes=20),
    "ROTATE_REFRESH_TOKENS": False,
    "BLACKLIST_AFTER_ROTATION": False,
    "ALGORITHM": "HS256",
    "SIGNING_KEY": SECRET_KEY,
    "UPDATE_LAST_LOGIN": True,
}

CORS_ALLOW_CREDENTIALS = True
CORS_ALLOWED_ORIGINS = [
    "http://127.0.0.1:5186",
    "http://localhost:5186",
    config("FRONTEND_URL", default=""),
]

# Add these for file uploads
CORS_ALLOW_HEADERS = [
    "accept",
    "accept-encoding",
    "authorization",
    "content-type",
    "dnt",
    "origin",
    "user-agent",
    "x-csrftoken",
    "x-requested-with",
]

CORS_ALLOW_METHODS = [
    "DELETE",
    "GET",
    "OPTIONS",
    "PATCH",
    "POST",
    "PUT",
]

# For development only - REMOVE this in production
CORS_ALLOW_ALL_ORIGINS = True
# Cloudflare R2 Configuration
CLOUDFLARE_R2_BUCKET = config("CLOUDFLARE_R2_BUCKET", cast=str, default="")
CLOUDFLARE_R2_ACCESS_KEY = config("CLOUDFLARE_R2_ACCESS_KEY", cast=str, default="")
CLOUDFLARE_R2_SECRET_KEY = config("CLOUDFLARE_R2_SECRET_KEY", cast=str, default="")
CLOUDFLARE_R2_BUCKET_ENDPOINT = config(
    "CLOUDFLARE_R2_BUCKET_ENDPOINT", cast=str, default=""
)

if CLOUDFLARE_R2_BUCKET:
    CLOUDFLARE_R2_CONFIG_OPTIONS = {
        "bucket_name": CLOUDFLARE_R2_BUCKET,
        "access_key": CLOUDFLARE_R2_ACCESS_KEY,
        "secret_key": CLOUDFLARE_R2_SECRET_KEY,
        "endpoint_url": CLOUDFLARE_R2_BUCKET_ENDPOINT,
        "default_acl": "public-read",
        "signature_version": "s3v4",
    }

    STORAGES = {
        "default": {
            "BACKEND": "helpers.cloudflare.storages.MediaFileStorage",
            "OPTIONS": CLOUDFLARE_R2_CONFIG_OPTIONS,
        },
        "staticfiles": {
            "BACKEND": "helpers.cloudflare.storages.StaticFileStorage",
            "OPTIONS": CLOUDFLARE_R2_CONFIG_OPTIONS,
        },
    }

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
    "loggers": {
        "django": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "APIBackend": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}
