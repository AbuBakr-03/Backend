from djoser.email import PasswordResetEmail
from django.conf import settings


class CustomPasswordResetEmail(PasswordResetEmail):
    template_name = "email/password_reset.html"

    def get_context_data(self):
        context = super().get_context_data()

        uid = context.get("uid")
        token = context.get("token")

        reset_url = f"http://127.0.0.1:5194/reset-password/{uid}/{token}"
        context["reset_url"] = reset_url
        return context
