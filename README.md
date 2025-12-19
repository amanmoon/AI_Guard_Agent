# AI Guard Agent

**Course:** EE 726 - Advanced Machine Learning (IIT Bombay)  
**Instructor:** Prof. Amit Sethi  
**Authors:** V Sathvik, Aman Moon  
**Status:** Active

## 📋 Abstract

The **AI Guard Agent** is an interactive security monitor developed as part of the **Advanced Machine Learning** curriculum. It integrates **Automatic Speech Recognition (ASR)**, **Facial Recognition**, and **Large Language Models (LLMs)** to create a system that secures environments by actively monitoring audio-visual inputs and verifying user identities.

It operates via a robust **multi-threaded Python backend**, ensuring real-time performance by handling blocking I/O operations (such as video feeds and audio streams) concurrently. The system utilizes OpenAI's Whisper for transcription, Google's Gemma for conversational intelligence, and dlib-based biometric models for security.

---

## 🛠️ Tech Stack & Libraries

The following libraries were utilized to build the various modules of the system:

| Library / Tool | Purpose & Implementation |
| :--- | :--- |
| **Eel** | Bridges the Python backend with the React/TypeScript frontend for asynchronous communication. |
| **OpenAI Whisper** | **ASR (Automatic Speech Recognition):** Converts raw audio input into text with high accuracy. |
| **Google Gemma** | **LLM (Large Language Model):** Provides the conversational intelligence (running on `torch` via Hugging Face). |
| **dlib** | **Face Detection & Recognition:** Uses HOG models to generate 128-d face encodings for verification. |
| **PyTorch (CUDA)** | **Inference Engine:** Accelerates the Gemma LLM and Whisper models using GPU computing. |
| **pyttsx3** | **TTS (Text-to-Speech):** Synthesizes the AI's text responses into spoken audio offline. |
| **NumPy** | **Signal Processing:** Performs Fast Fourier Transforms (FFT) to generate real-time audio visualizations. |
| **FuzzyWuzzy / Jellyfish** | **Command Matching:** Implements Levenshtein distance and phonetic matching for command recognition. |
| **OpenCV** | **Video Capture:** Manages the webcam feed and frame extraction for the vision thread. |
| **React + TypeScript** | **Frontend UI:** Provides a dynamic, type-safe user interface for the security dashboard. |

---

## 🏗️ System Architecture & Multi-Threading

The core strength of the AI Guard Agent lies in its **concurrent execution architecture**. To ensure low latency and prevent the main application loop from freezing during heavy inference or I/O tasks, the system implements a **multi-threaded architecture** consisting of three primary worker threads:

### 1. Audio Processor Thread (Daemon)
* **Role:** Handles continuous audio ingestion and processing.
* **Functionality:**
    * Captures raw PCM data at 16kHz.
    * Performs **Voice Activity Detection (VAD)** to filter silence and ambient noise.
    * Computes **Fast Fourier Transform (FFT)** for real-time frequency visualization on the UI.
    * Pushes valid audio segments to the recognition queue.

### 2. Vision Engine Thread (Daemon)
* **Role:** Manages the video feed and biometric verification.
* **Functionality:**
    * Operates on a **Producer-Consumer model** to decouple frame capture from processing.
    * Uses **dlib's HOG face detector** to locate faces in real-time.
    * Generates 128-dimensional face encodings and compares them against a trusted database using Euclidean distance.
    * Updates the global `is_verified` state flag based on temporal smoothing.

### 3. Inference Engine Thread
* **Role:** The "Brain" of the agent.
* **Functionality:**
    * Manages the **Large Language Model (Google Gemma)** context window.
    * Receives transcribed text from the Audio thread.
    * Generates context-aware responses (switching between "Assistant" and "Guard" personas).
    * Synthesizes speech output using `pyttsx3` (TTS).

---

## 🎤 Audio Subsystem Implementation

The audio logic prioritizes responsiveness and visualization.

* **Signal Processing:** Raw audio is decomposed into frequency bins (50Hz - 16kHz). Logarithmic transformation is applied to magnitudes to approximate human loudness perception for the UI visualizer:
    $$\text{log(magnitudes)} = \ln(1 + \text{band(magnitudes)})$$
* **Voice Activity Detection (VAD):** A dynamic silence threshold is established via calibration:
    $$T_{silence} = \max(100, A_{ambient\_max} \times 1.2)$$
* **Command Recognition:** Uses fuzzy string matching to detect activation keywords with >85% confidence.

---

## 👁️ Computer Vision Module

The vision module ensures that only authorized personnel can access the system's "Helpful" persona.

* **Enrollment:** Captures trusted faces and stores their 128-d encodings.
* **Verification Logic:**
    $$||E_{live} - E_{trusted}|| < 0.6$$
    If the Euclidean distance is below 0.6, the face is verified.
* **Anti-Spoofing/Stability:** Implementation includes a temporal buffer to prevent flickering decisions due to motion blur or lighting changes.

---

## 🧠 Generative AI Integration

The conversational core utilizes **Google's Gemma model** (`google/gemma-3n-E4b-it`), optimized for consumer hardware using `torch.bfloat16` precision.

### Context-Aware Prompt Engineering
The agent's behavior is dictated by dynamic prompt injection:

1.  **Verified State:** "You are a helpful security assistant. Answer the user's queries concisely."
2.  **Unverified State (Guard Mode):** > "Instructions: You are a guard AI. An unverified user is trying to talk to you. Politely but firmly, tell them to please kindly leave the room."

---

## 🚀 Installation & Usage

### Prerequisites
* **Python 3.8+**
* **Node.js & npm** (Required for building the frontend)
* **CUDA-enabled GPU** (Recommended for Gemma LLM)
* Webcam and Microphone

### Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/amanmoon/AI_Guard_Agent.git](https://github.com/amanmoon/AI_Guard_Agent.git)
    cd AI_Guard_Agent
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install Node dependencies:**
    Navigate to the root directory where `package.json` is located and install the frontend packages.
    ```bash
    npm install
    ```

4.  **Build the Frontend:**
    This compiles the React/TypeScript code into static files that Eel can serve.
    ```bash
    npm run build
    ```

5.  **Run the Application:**
    Once the build is complete, start the Python backend.
    ```bash
    python eelApplication.py
    ```

---

## 🔄 Integration Flow

1.  **Idle Mode:** System listens for the specific "Activate" keyword.
2.  **Active Mode:** * **Vision Thread** continuously checks for faces.
    * **Audio Thread** captures user speech.
    * **Whisper** transcribes speech -> Text.
    * **LLM** checks `is_verified` flag -> Generates Response.
    * **TTS** speaks the response back to the user.

---

© 2025 AI Guard Agent Project | IIT Bombay
