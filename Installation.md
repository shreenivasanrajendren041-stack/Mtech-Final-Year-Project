# Installation and Setup Guide

This document provides complete instructions for setting up, configuring, and running the **Identity-Aware Cross-Device Multi-Factor Authentication** system on your local workstation.

---

## Prerequisites

Before starting the installation, ensure your system meets the following requirements:
* **Operating System:** Windows 11 / Linux / macOS
* **Python Version:** Python 3.10 or higher
* **Hardware:** Webcam enabled for live facial biometrics and anti-fraud mask detection.

---

## Step 1: Clone the Repository

Clone the project repository from GitHub to your local machine:

```bash
git clone https://github.com/your-username/Identity-Aware-MFA-Auth.git
cd Identity-Aware-MFA-Auth
```

---

## Step 2: Set Up a Python Virtual Environment

It is recommended to use a virtual environment to manage dependencies cleanly:

```bash
# Create the virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows (Command Prompt / PowerShell):
venv\Scripts\activate

# On macOS / Linux:
source venv/bin/activate
```

---

## Step 3: Install Required Dependencies

Install all required Python packages for Flask, PyTorch, ArcFace embeddings, OpenCV, and computer vision utilities:

```bash
pip install -r requirements.txt
```

*(Note: If `requirements.txt` is not yet generated, ensure core packages including `Flask`, `torch`, `torchvision`, `opencv-python`, `numpy`, and `scikit-learn` are installed).*

---

## Step 4: Directory Structure Setup

Ensure your project root folder contains the following directory structure so the application can load datasets and templates correctly:

```text
Identity-Aware-MFA-Auth/
│
├── dataset/             # Storage for registered user facial embeddings/images
├── static/              # CSS stylesheets, icons, and frontend assets
├── templates/           # HTML templates (dashboard.html, deleteusers.html, etc.)
├── app.py               # Main Flask backend server
├── deleted_users.json   # Persistent blocklist for revoked accounts
└── INSTALLATION.md      # Setup documentation
```

---

## Step 5: Run the Application

Start the Flask development server:

```bash
python app.py
```

* Open your web browser and navigate to: **`http://127.0.0.1:5000`**
* **Admin Access:** Log in using the designated owner account (`pers1`) to unlock full administrative privileges, view the directory, and manage user deletions.
