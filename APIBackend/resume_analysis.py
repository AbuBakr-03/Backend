# APIBackend/resume_analysis.py
import os
import json
from PyPDF2 import PdfReader
import google.generativeai as genai
from django.conf import settings
import logging
from django.core.files.storage import default_storage

logger = logging.getLogger(__name__)


class ResumeAnalysisService:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning(
                "No Gemini API key found. Question generation will not work."
            )
            return

        # Configure the Gemini API
        genai.configure(api_key=self.api_key)

        # Use Gemini Pro model
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    def extract_text_from_pdf(self, pdf_path):
        """Extract text content from a PDF file."""
        try:
            # Use default_storage.open() instead of open() for cloud storage
            with default_storage.open(pdf_path, "rb") as file:
                pdf_reader = PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""

    def generate_interview_questions(self, resume_text, job_description=""):
        """Generate interview questions based on the resume text using Gemini API."""
        if not hasattr(self, "model"):
            logger.error("Gemini API not properly initialized.")
            return None

        prompt = f"""
        You are an expert interviewer.

        Based on the following resume and job description, generate 15-20 highly relevant interview questions.

        - Focus on assessing the candidate's technical skills, educational background, and professional experience.
        - Include both behavioral and technical questions relevant to the job.
        - Questions should be customized based on the actual skills, projects, and education mentioned.
        - Avoid generic or irrelevant questions.
        - If the job requires specific technical skills, include questions to test those skills.

        Format your response as a JSON array of question objects with categories.

        Resume Content:
        {resume_text}

        Job Description:
        {job_description}

        Return ONLY the JSON array of questions in this format:
        [
            {{
                "category": "Technical Skill",
                "question": "Question text here"
            }},
            ...
        ]
        """

        try:
            # Generate content with Gemini
            response = self.model.generate_content(prompt)
            response_text = response.text

            # Parse JSON response
            if "```json" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_text = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_text = response_text

            # Parse the JSON
            questions_data = json.loads(json_text)
            return questions_data

        except Exception as e:
            logger.error(f"API request failed: {e}")
            return None

    def analyze_resume_for_job(self, resume_path, job_description):
        resume_text = self.extract_text_from_pdf(resume_path)
        if not resume_text:
            logger.warning("Failed to extract text from resume.")
            return None

        return self.generate_interview_questions(resume_text, job_description)
