# SmartHR Backend API

A comprehensive Django REST Framework backend for an intelligent HR management system with AI-powered interview analysis and resume screening capabilities.

## 🚀 Features

- **REST API** - Complete REST endpoints for HR operations
- **AI-Powered Interview Analysis** - Audio and facial emotion recognition using TensorFlow/Keras
- **Resume Screening** - Automated resume analysis and scoring using NLP
- **Interview Question Generation** - AI-generated interview questions using Google Gemini API
- **Role-based Authentication** - JWT authentication with recruiter/admin roles
- **Video Interview Analysis** - Real-time emotion detection from interview recordings
- **Email Integration** - Password reset and notifications via Resend SMTP
- **Media File Handling** - Resume and video file upload support

## 🛠️ Tech Stack

- **Django 4.x** with Django REST Framework
- **PostgreSQL** database
- **TensorFlow/Keras** for machine learning models
- **OpenCV** for computer vision tasks
- **Librosa** for audio processing
- **NLTK/scikit-learn** for NLP and text analysis
- **Google Gemini API** for AI-powered question generation
- **JWT Authentication** with SimpleJWT
- **Resend SMTP** for email services
- **CORS** enabled for frontend integration

## 📦 Installation

1. Clone the repository
```bash
git clone <repository-url>
cd backend
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up environment variables
```bash
# Create .env file
DATABASE_URL=postgresql://username:password@localhost:5432/smarthr_db
GEMINI_API_KEY=your_gemini_api_key_here
```

5. Run migrations
```bash
python manage.py makemigrations
python manage.py migrate
```

6. Create a superuser
```bash
python manage.py createsuperuser
```

7. Start the development server
```bash
python manage.py runserver
```

The API will be available at `http://127.0.0.1:8000`

## 🗃️ Database Setup

Configure PostgreSQL database and update the `DATABASE_URL` environment variable:

```
DATABASE_URL=postgresql://user:password@host:port/database_name
```

## 🤖 AI Models Setup

The system requires pre-trained models for interview analysis:

1. **Audio Emotion Model** - `APIBackend/AImodels/full_audio_emotion_model.h5`
2. **Face Expression Model** - `APIBackend/AImodels/face_expression_model3.h5`
3. **Preprocessing Files** - `scaler2.pickle` and `encoder2.pickle`

Place these files in the `APIBackend/AImodels/` directory.

## 🔧 Configuration

### Environment Variables
- `DATABASE_URL` - PostgreSQL connection string
- `GEMINI_API_KEY` - Google Gemini API key for question generation
- `DEBUG` - Set to `False` in production

### Email Configuration
The system uses Resend SMTP for email services. Update settings in `settings.py`:
- `EMAIL_HOST_PASSWORD` - Your Resend API key
- `DEFAULT_FROM_EMAIL` - Sender email address

## 📚 API Endpoints

### Authentication
- `POST /auth/jwt/create/` - Login and get JWT tokens
- `POST /auth/jwt/refresh/` - Refresh access token
- `POST /auth/jwt/logout/` - Logout and blacklist token
- `POST /auth/users/` - User registration
- `POST /auth/users/reset_password/` - Password reset

### Core HR Endpoints
- `GET/POST /api/company/` - Company management
- `GET/POST /api/department/` - Department management
- `GET/POST /api/job/` - Job postings
- `GET/POST /api/application/` - Job applications
- `GET/POST /api/interview/` - Interview scheduling

### AI-Powered Features
- `POST /api/interview/{id}/analyze-recording/` - Analyze interview video
- `POST /api/interview/{id}/generate-questions/` - Generate interview questions
- `GET /api/predicted-candidates/` - AI-evaluated candidates

### Admin Features
- `GET/POST /api/groups/recruiters/` - Recruiter role management
- `GET /api/recruiters/` - Recruiter requests

## 🔐 Authentication & Permissions

The system implements role-based access control:

- **Superusers** - Full system access
- **Recruiters** - Can manage jobs, applications, and interviews
- **Regular Users** - Can apply for jobs and participate in interviews

JWT tokens include role information for frontend role-based UI rendering.

## 🎯 Key Services

### Interview Analysis Service
- **Audio Emotion Recognition** - Analyzes candidate's emotional state from voice
- **Facial Expression Analysis** - Detects emotions from video frames
- **Confidence Scoring** - Generates overall interview performance scores

### Resume Screening Service
- **Text Extraction** - Extracts content from PDF resumes
- **Match Scoring** - Calculates compatibility with job descriptions
- **Keyword Analysis** - Identifies relevant skills and experience

### Question Generation Service
- **AI-Powered** - Uses Google Gemini API for intelligent question generation
- **Context-Aware** - Tailors questions based on resume content and job requirements
- **Categorized Output** - Groups questions by technical/behavioral categories

