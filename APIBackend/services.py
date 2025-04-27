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

# Make sure to download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


class ResumeScreeningService:
    def __init__(self):
        self.stopwords_path = os.path.join(
            settings.BASE_DIR, "APIBackend/data/stopwords.txt"
        )
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(self.stopwords_path), exist_ok=True)
        # Create a simple stopwords list if file doesn't exist
        if not os.path.exists(self.stopwords_path):
            with open(self.stopwords_path, "w") as f:
                stopwords = nltk.corpus.stopwords.words("english")
                f.write("\n".join(stopwords))

    def read_resume(self, resume_path):
        try:
            with open(resume_path, "rb") as f:
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
        # Convert text to lowercase
        cleaned_str = str(text).lower()
        # Remove web addresses
        cleaned_str = re.sub(r"(http://\S*)", "", cleaned_str)
        cleaned_str = re.sub(r"(https://\S*)", "", cleaned_str)
        # Remove email addresses
        cleaned_str = re.sub(
            "([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+\\S*)", "", cleaned_str
        )
        # Clean slashes and other punctuation
        for char in ["/", ";", ":", ".", ",", "-"]:
            cleaned_str = re.sub(re.escape(char), " ", cleaned_str)
        # Remove punctuations
        exclude = set(string.punctuation)
        cleaned_str = "".join(ch for ch in cleaned_str if ch not in exclude)
        # Remove numbers/digits
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

        # Using lemmatization
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

        # Calculate percentage match
        if len(jd_text) > 0:
            match_percentage = (len(matched_words) / len(jd_text)) * 100
        else:
            match_percentage = 0

        # Return match score and matched words
        return match_percentage, matched_words

    def screen_resume(self, resume_path, job):
        resume_content = self.read_resume(resume_path)
        job_description, stemmed_words_dict = self.process_job_description(job)

        match_score, matched_words = self.calculate_match_score(
            resume_content, job_description, stemmed_words_dict
        )

        # Determine status based on match score
        # ID 1: Pending, ID 2: Approved for Interview, ID 3: Rejected
        if match_score >= 50:  # Adjust threshold as needed
            status_id = 2  # Approved for interview
        else:
            status_id = 3  # Rejected

        return {
            "match_score": match_score,
            "status_id": status_id,
            "matched_words": matched_words,
        }
