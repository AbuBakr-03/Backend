# APIBackend/services.py
import os
import re
import string
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer
from PyPDF2 import PdfReader
from django.conf import settings
from django.core.files.storage import default_storage

# Make sure to download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


class ResumeScreeningService:
    def __init__(self):
        self.stopwords_path = os.path.join(
            settings.BASE_DIR, "APIBackend/data/stopwords.txt"
        )
        os.makedirs(os.path.dirname(self.stopwords_path), exist_ok=True)
        if not os.path.exists(self.stopwords_path):
            with open(self.stopwords_path, "w") as f:
                stopwords = nltk.corpus.stopwords.words("english")
                f.write("\n".join(stopwords))

    def read_resume(self, resume_path):
        try:
            # Use default_storage.open() instead of open() for cloud storage
            with default_storage.open(resume_path, "rb") as f:
                pdf_reader = PdfReader(f)
                content = "\n".join(
                    page.extract_text().strip() for page in pdf_reader.pages
                )
                content = " ".join(content.split())
                return self.clean_text(content, False)
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""

    def process_job_description(self, job):
        # Combine responsibilities, qualifications, and nice_to_haves
        job_description = (
            f"{job.responsiblities} {job.qualification} {job.nice_to_haves}"
        )
        return self.clean_text(job_description, True)

    def clean_text(self, text, is_jd_file):
        cleaned_str = str(text).lower()
        cleaned_str = re.sub(r"(http://\S*)", "", cleaned_str)
        cleaned_str = re.sub(r"(https://\S*)", "", cleaned_str)
        cleaned_str = re.sub(
            "([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+\\S*)", "", cleaned_str
        )
        for char in ["/", ";", ":", ".", ",", "-"]:
            cleaned_str = re.sub(re.escape(char), " ", cleaned_str)
        exclude = set(string.punctuation)
        cleaned_str = "".join(ch for ch in cleaned_str if ch not in exclude)
        cleaned_str = re.sub("\\d", "", cleaned_str)

        return self.stem_and_remove_stopwords(cleaned_str, is_jd_file)

    def stem_and_remove_stopwords(self, txt_str, is_jd_file):
        stemmed_words_dict = {}
        tokens = txt_str.split()

        try:
            with open(self.stopwords_path, "r") as f:
                stopwords_list = [line.strip() for line in f]
        except:
            stopwords_list = nltk.corpus.stopwords.words("english")

        clean_str = ""
        for t in tokens:
            if t in stopwords_list or len(t) <= 1:
                continue
            else:
                clean_str = clean_str + " " + t
        clean_str = clean_str.strip()

        lemma = nltk.wordnet.WordNetLemmatizer()
        cleaned_str = " ".join([lemma.lemmatize(s) for s in clean_str.split(" ")])
        cleaned_str = " ".join(
            [LancasterStemmer().stem(s) for s in cleaned_str.split(" ")]
        )

        if is_jd_file:
            for s in clean_str.split(" "):
                stemmed_words_dict[PorterStemmer().stem(s)] = s
            return cleaned_str, stemmed_words_dict
        else:
            return cleaned_str

    def calculate_match_score(
        self, resume_content, job_description, stemmed_words_dict
    ):
        # Tokenize resume content and job description
        resume_text = list(set(resume_content.split()))
        jd_text = list(set(job_description.split()))

        # Count matches
        matched_words = [val for val in jd_text if val in resume_text]

        if len(jd_text) > 0:
            match_percentage = (len(matched_words) / len(jd_text)) * 100
        else:
            match_percentage = 0

        return match_percentage, matched_words

    def screen_resume(self, resume_path, job):
        resume_content = self.read_resume(resume_path)
        job_description, stemmed_words_dict = self.process_job_description(job)

        match_score, matched_words = self.calculate_match_score(
            resume_content, job_description, stemmed_words_dict
        )

        # ID 1: Pending, ID 2: Approved for Interview, ID 3: Rejected
        if match_score >= 50:
            status_id = 2  # Approved for interview
        else:
            status_id = 3  # Rejected

        return {
            "match_score": match_score,
            "status_id": status_id,
            "matched_words": matched_words,
        }


class CandidateEvaluationService:
    """Service for managing candidate evaluations"""

    @staticmethod
    def process_interview_result(interview):
        """
        Process an interview result and create a predicted candidate if necessary
        Returns a predicted candidate if created, otherwise None
        """
        from .models import PredictedCandidate, EvaluationStatus

        # Only process interviews with "pass" result (ID 2)
        if interview.result.id != 2:
            return None

        # Check if a predicted candidate already exists
        try:
            predicted_candidate = interview.predicted_candidate
            return predicted_candidate  # Already exists
        except PredictedCandidate.DoesNotExist:
            # Create new predicted candidate with pending status
            status = EvaluationStatus.objects.get(id=1)  # 1 = Pending
            predicted_candidate = PredictedCandidate.objects.create(
                interview=interview, status=status
            )
            return predicted_candidate

    @staticmethod
    def evaluate_candidate(predicted_candidate, evaluation_data):
        """
        Evaluate a predicted candidate based on form responses
        Returns the updated candidate
        """
        # Calculate average score
        questions = evaluation_data.get("questions", [])
        if not questions:
            return predicted_candidate

        total_score = 0
        question_count = len(questions)

        for question in questions:
            # Each question has a score from 1-5
            score = question.get("score", 0)
            total_score += score

        # Calculate average (0-5 scale)
        average_score = total_score / question_count if question_count > 0 else 0

        # Determine status based on score (>= 3.5 out of 5 is a pass)
        status_id = 2 if average_score >= 3.5 else 3  # 2 = Hired, 3 = Rejected

        # Update the candidate
        from .models import EvaluationStatus

        predicted_candidate.evaluation_data = evaluation_data
        predicted_candidate.evaluation_score = average_score
        predicted_candidate.status = EvaluationStatus.objects.get(id=status_id)
        predicted_candidate.save()

        return predicted_candidate
