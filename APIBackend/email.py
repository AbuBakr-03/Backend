# APIBackend/email.py
from djoser.email import PasswordResetEmail
from django.conf import settings


class CustomPasswordResetEmail(PasswordResetEmail):
    template_name = "email/password_reset.html"  # We'll create this template

    def get_context_data(self):
        context = super().get_context_data()

        # Override the reset URL with our frontend URL
        user = context.get("user")
        uid = context.get("uid")
        token = context.get("token")

        # Generate the complete URL to the frontend
        reset_url = f"http://127.0.0.1:5177/reset-password/{uid}/{token}"
        context["reset_url"] = reset_url

        return context
