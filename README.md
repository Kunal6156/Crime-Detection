# 🚨 Multimodal Crime Detection System
This project implements a comprehensive **Multimodal Crime Detection System** capable of analyzing images, audio, and video content for suspicious activities and potential crimes. Leveraging advanced Large Language Models (LLMs) and robust media processing techniques, the system aims to provide accurate, timely, and actionable insights for security and monitoring applications. It features intelligent caching, asynchronous processing, cross-provider analysis, and optional local LLM verification for enhanced reliability.

-----

## ✨ Features

  * **🖼️ Image Analysis**: Detects crimes and suspicious activities in images using powerful LLMs.
  * **🔊 Audio Analysis**: Transcribes audio and identifies distress signals (screams, sirens) and other suspicious sounds.
  * **📹 Video Analysis**: Processes video content by extracting and analyzing frames and audio tracks, providing a holistic view of potential incidents.
  * **🧠 Multi-LLM Support**: Integrates with various LLM providers (Google Gemini, OpenAI, Anthropic, Replicate, HuggingFace) for flexible and resilient analysis.
  * **🏡 Local LLM Verification**: Optional on-premise LLM for cross-validating cloud-based LLM results, enhancing confidence and reducing false positives.
  * **⚡ Asynchronous Processing**: Queues long-running tasks (like video analysis) to provide immediate responses, with webhook callbacks for results.
  * **📊 Intelligent Caching**: Caches analysis results for frequently processed media to improve efficiency and reduce API costs.
  * **📧 Notification System**: Configurable email alerts for detected crimes, with severity-based recipient routing.
  * **📈 Detailed Reporting**: Generates comprehensive reports of analysis results for easy review and record-keeping.
  * **📂 Batch & Directory Analysis**: Analyze multiple files or entire directories for efficient large-scale processing.
  * **Robust Error Handling**: Comprehensive logging and error management across all modules.

-----

## 🛠️ Installation

### Prerequisites

  * Python 3.8+
  * `ffmpeg` (for video and audio processing)
  * `git` (for cloning the repository)

### Steps

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Kunal260204/Multimodal-Crime-Detection-System.git
    cd Multimodal-Crime-Detection-System
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    *(Note: The `requirements.txt` file is not provided but is assumed to contain `flask`, `flask-cors`, `werkzeug`, `requests`, `Pillow`, `opencv-python`, `moviepy`, `librosa`, `SpeechRecognition`, `pydub`, `numpy`, `anthropic`, `openai`, `google-generativeai`, `langchain-community`, `transformers`, `sentence-transformers`, `torch`, and `ctransformers` if local LLMs are used.)*

4.  **Configure API Keys:**

    Create a `config.json` file in the project root with your LLM API keys and optional notification settings. Example:

    ```json
    {
      "gemini": "YOUR_GEMINI_API_KEY",
      "openai": "YOUR_OPENAI_API_KEY",
      "anthropic": "YOUR_ANTHROPIC_API_KEY",
      "replicate": "YOUR_REPLICATE_API_TOKEN",
      "huggingface": "YOUR_HUGGINGFACE_API_TOKEN",
      "default_provider": "gemini",
      "notification": {
        "smtp_server": "smtp.your-email.com",
        "port": 465,
        "username": "your_email@example.com",
        "password": "your_email_password",
        "sender_email": "crime.alert.system@example.com",
        "default_recipients": ["security_team@example.com"],
        "notify_threshold": "medium"
      }
    }
    ```

    **Important Security Note**: Never commit your `config.json` file directly to a public repository. Use environment variables or a `.env` file for production deployments.

5.  **Setup Uploads and Temp Folders:**

    The system will automatically create `uploads` and `temp` directories in the project root. These are used for storing uploaded files and temporary media processing outputs (like video frames and extracted audio).

-----

## 🚀 Usage

### Running the API Server

The system exposes a RESTful API for integration.

```bash
python main.py
```

The API server will start on `http://0.0.0.0:5000` by default.

### API Endpoints

All API requests expect `Content-Type: application/json` for POST requests (except `/upload`).

#### 1\. Health Check

Checks the status of the API server and its components.

  * **GET** `/health`
  * **Response:**
    ```json
    {
        "status": "healthy",
        "timestamp": "2023-10-27T10:30:00.000000",
        "providers": ["gemini", "openai"],
        "notification_enabled": true
    }
    ```

#### 2\. Get Available LLM Providers

Lists the configured LLM providers.

  * **GET** `/providers`
  * **Response:**
    ```json
    {
        "providers": ["gemini", "openai", "anthropic"],
        "default_provider": "gemini"
    }
    ```

#### 3\. Upload File

Uploads a media file to the server for subsequent analysis.

  * **POST** `/upload`
  * **Content-Type**: `multipart/form-data`
  * **Form Data**: `file` (the actual media file)
  * **Example (using `curl`):**
    ```bash
    curl -X POST -F "file=@/path/to/your/image.jpg" http://localhost:5000/upload
    ```
  * **Response:**
    ```json
    {
        "status": "success",
        "file_path": "uploads/image.jpg",
        "filename": "image.jpg"
    }
    ```

#### 4\. Analyze Image from Path

Analyzes an image file already present on the server (e.g., uploaded via `/upload`).

  * **POST** `/analyze/image/path`
  * **Request Body:**
    ```json
    {
        "image_path": "uploads/suspect_image.png",
        "provider": "openai",  // Optional, uses default if not specified
        "async": false,        // Optional, default is false
        "callback_url": "http://your-webhook.com/callback", // Optional, for async mode
        "notify": false        // Optional, sends email notification if crime detected and notifier is configured
    }
    ```
  * **Response (Synchronous):**
    ```json
    {
        "crime_detected": true,
        "crime_type": "Theft/Burglary",
        "confidence": "high",
        "evidence_description": "Person attempting to force open a locked door...",
        "image_path": "uploads/suspect_image.png",
        "local_analysis": { ... }, // if local LLM is enabled
        "cross_verification": { ... }, // if local LLM is enabled
        "timestamp": "2023-10-27T10:35:00.000000",
        "source": "api",
        "provider": "openai"
    }
    ```
  * **Response (Asynchronous):**
    ```json
    {
        "status": "queued",
        "job_id": "image_file_1678888888_abc123",
        "media_type": "image_file",
        "timestamp": "2023-10-27T10:35:00.000000"
    }
    ```

#### 5\. Analyze Image from URL

Analyzes an image directly from a provided URL.

  * **POST** `/analyze/image/url`
  * **Request Body:**
    ```json
    {
        "image_url": "http://example.com/suspicious_activity.jpg",
        "provider": "gemini",
        "async": false,
        "callback_url": "http://your-webhook.com/callback",
        "notify": false
    }
    ```
  * **Response:** (Similar to `analyze/image/path`)

#### 6\. Analyze Audio

Analyzes an audio file from a path on the server.

  * **POST** `/analyze/audio`
  * **Request Body:**
    ```json
    {
        "audio_path": "uploads/suspicious_audio.mp3",
        "provider": "openai",
        "async": false,
        "callback_url": "http://your-webhook.com/callback",
        "notify": false
    }
    ```
  * **Response (Synchronous):**
    ```json
    {
        "crime_detected": true,
        "crime_type": "Assault/Violence",
        "confidence": "high",
        "evidence_description": "Audio contains sounds of shouting and a distinct scream.",
        "transcription": "[POSSIBLE SIRENS OR SCREAMS DETECTED]",
        "audio_path": "uploads/suspicious_audio.mp3",
        "local_analysis": { ... },
        "cross_verification": { ... },
        "timestamp": "2023-10-27T10:40:00.000000",
        "source": "api",
        "provider": "openai"
    }
    ```
  * **Response (Asynchronous):**
    ```json
    {
        "status": "queued",
        "job_id": "audio_file_1678888890_def456",
        "media_type": "audio_file",
        "timestamp": "2023-10-27T10:40:00.000000"
    }
    ```

#### 7\. Analyze Video

Analyzes a video file from a path on the server. Video analysis is **asynchronous by default** due to its longer processing time.

  * **POST** `/analyze/video`
  * **Request Body:**
    ```json
    {
        "video_path": "uploads/security_footage.mp4",
        "provider": "gemini",        // Optional, uses default if not specified
        "frame_interval": 5,         // Optional, analyze every 5 seconds (default is 5)
        "async": true,               // Optional, default is true for video
        "callback_url": "http://your-webhook.com/callback",
        "notify": false
    }
    ```
  * **Response (Asynchronous - default):**
    ```json
    {
        "status": "queued",
        "job_id": "video_file_1678888892_ghi789",
        "media_type": "video_file",
        "timestamp": "2023-10-27T10:45:00.000000"
    }
    ```
  * **Response (Synchronous - if `async: false` is explicitly set, not recommended for large videos):**
    ```json
    {
        "crime_detected": true,
        "cross_validation_results": { ... },
        "provider_results": [ { ... } ],
        "frame_results": { ... },
        "audio_result": { ... },
        "local_analysis": { ... },
        "video_path": "uploads/security_footage.mp4",
        "timestamp": "2023-10-27T10:46:00.000000"
    }
    ```

#### 8\. Send Manual Notification

Manually trigger an email notification based on a provided detection result.

  * **POST** `/notify`
  * **Request Body:**
    ```json
    {
        "detection_result": {
            "crime_detected": true,
            "crime_type": "Vandalism/Property Damage",
            "confidence": "high",
            "potential_severity": "medium",
            "evidence_description": "Graffiti being sprayed on a wall.",
            "recommendations": "Dispatch security to investigate."
        },
        "media_path": "uploads/graffiti_image.jpg" // Optional path to media for context
    }
    ```
  * **Response:**
    ```json
    {
        "status": "sent",
        "recipients": ["security_team@example.com"],
        "crime_type": "Vandalism/Property Damage",
        "severity": "medium",
        "timestamp": "2023-10-27T10:50:00.000000"
    }
    ```

#### 9\. Get Job Status

Retrieves the status and results of an asynchronously queued job.

  * **GET** `/jobs/<job_id>`
  * **Example:** `/jobs/image_file_1678888888_abc123`
  * **Response (Pending):**
    ```json
    {
        "status": "pending",
        "job_id": "image_file_1678888888_abc123"
    }
    ```
  * **Response (Completed):**
    ```json
    {
        "status": "success",
        "job_id": "image_file_1678888888_abc123",
        "crime_detected": true,
        "crime_type": "Theft/Burglary",
        "confidence": "high",
        // ... full analysis result ...
    }
    ```

#### 10\. Serve Uploaded Files

Allows direct access to uploaded files for viewing.

  * **GET** `/uploads/<filename>`
  * **Example:** `/uploads/suspect_image.png`

-----

## 💻 Project Structure

```
.
├── app/
│   ├── __init__.py         # Package initialization, logging, and global config
│   ├── api.py              # Flask API routes and server setup
│   ├── crime_detection.py  # Core crime detection pipeline logic
│   ├── llm_manager.py      # Manages interactions with different LLM providers
│   ├── media_processor.py  # Handles all media (image, audio, video) loading, preprocessing, and feature extraction
│   ├── notification.py     # Email notification system for alerts
│   └── utils/
│       └── cache.py        # Caching mechanism for analysis results (assumed to be here)
│       └── rate_limiter.py # Rate limiting for API calls (assumed to be here)
├── main.py                 # Entry point for the application
├── config.json             # API keys and system configuration (user-defined)
├── requirements.txt        # Python dependencies
├── uploads/                # Directory for uploaded media files (created by the app)
└── temp/                   # Directory for temporary media processing files (e.g., video frames, audio extractions)
```

-----

## ⚙️ Configuration

The `config.json` file is crucial for setting up the system.

  * **LLM API Keys**: Provide your API keys for the desired LLM providers. At least one LLM provider must be configured.
      * `"gemini"`: Google Gemini API Key
      * `"openai"`: OpenAI API Key
      * `"anthropic"`: Anthropic API Key
      * `"replicate"`: Replicate API Token
      * `"huggingface"`: HuggingFace API Token
      * `"default_provider"`: Specifies which LLM to use by default if not explicitly provided in a request.
  * **Notification Settings (Optional)**:
      * `"smtp_server"`: Your SMTP server address (e.g., `smtp.gmail.com`).
      * `"port"`: SMTP server port (e.g., `465` for SSL, `587` for TLS).
      * `"username"`: Email account username for sending notifications.
      * `"password"`: Email account password. **Use an app-specific password if using Gmail or similar services.**
      * `"sender_email"`: The email address from which notifications will be sent.
      * `"default_recipients"`: A list of email addresses to receive all notifications.
      * `"notify_threshold"`: Minimum severity level (`"low"`, `"medium"`, `"high"`) at which notifications should be sent.
  * **Local LLM (Advanced)**:
      * The `CrimeDetectionPipeline` constructor takes `use_local_llm` and `local_llm_size` parameters. These are set within `api.py` when initializing the pipeline.
      * `local_llm_size`: `"tiny"`, `"medium"`, or `"large"` determines which local model to load (if `use_local_llm` is `True`).
      * Local LLM setup requires additional dependencies like `torch`, `transformers`, `sentence-transformers`, and `ctransformers` for GGUF models.

-----

## 🤝 Contributing

We welcome contributions to enhance the Multimodal Crime Detection System\!

1.  **Fork** the repository.
2.  **Create** a new branch (`git checkout -b feature/your-feature-name`).
3.  **Implement** your changes.
4.  **Write** tests for your new features.
5.  **Ensure** all existing tests pass.
6.  **Commit** your changes (`git commit -m 'feat: Add new feature'`).
7.  **Push** to the branch (`git push origin feature/your-feature-name`).
8.  **Open** a Pull Request.

-----

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

-----

## 📞 Contact

For any questions or inquiries, please open an issue on the GitHub repository.

-----
