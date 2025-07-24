from djoser.email import PasswordResetEmail
from django.conf import settings


class CustomPasswordResetEmail(PasswordResetEmail):
    template_name = "email/password_reset.html"

    def get_context_data(self):
        context = super().get_context_data()

        # Safely get uid and token with fallback
        uid = context.get("uid")
        token = context.get("token")

        # Only build reset_url if both uid and token exist
        if uid and token:
            reset_url = f"http://127.0.0.1:5173/reset-password/{uid}/{token}"
            context["reset_url"] = reset_url
        else:
            # Fallback for debugging
            context["reset_url"] = (
                "http://127.0.0.1:5173/reset-password/MISSING_UID/MISSING_TOKEN"
            )

        return context
