# 📷 Camera Movement Detector

This project is a user-friendly web application that detects **camera movements** (translation, rotation, zoom) in a video file and visualizes them. The algorithm analyzes the motion of the camera itself (pan, tilt, shake, zoom), not the movement of objects within the scene.

## Features

- **Automatic Feature Matching:** ORB-based keypoint matching and homography analysis.
- **Movement Type Separation:** Detects translation, rotation, and zoom independently.
- **Advanced Filtering:** Uses inlier ratio and ratio test to filter out object motion within the scene.
- **User-Friendly Interface:** Streamlit-based web app for uploading videos and visually inspecting results.
- **Easy Deployment:** Can be easily deployed on platforms like Streamlit Cloud or Hugging Face Spaces.

## Installation

1. **Requirements**
    - Python 3.8+
    - `pip install -r requirements.txt`

2. **Run Locally**
    ```bash
    pip install -r requirements.txt
    streamlit run app.py
    ```
    Then open `http://localhost:8501` in your browser.

## Usage

1. **Upload Video:** Select a video file (`.mp4`, `.mov`, `.avi`) in the interface.
2. **Configure Settings:** Adjust movement thresholds and advanced parameters from the sidebar.
3. **Analyze:** Click the "Detect Movement" button.
4. **Review Results:** Detected movement frames and their types are shown in detail.

## Deployment

### Streamlit Cloud

1. Upload all files to a GitHub repository.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and start a new app.
3. Select `app.py` as the main file and deploy.
App is online now: https://movementdetector-bkvmkpkkj4bsajfkjyd4rc.streamlit.app/

### Hugging Face Spaces

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and create a new Space.
2. Choose **Streamlit** as the SDK.
3. Upload your project files or link from GitHub.
4. Start the deployment.

## File Descriptions

- `app.py` : Main Streamlit web interface.
- `movement_detector.py` : Camera movement detection algorithm (class-based).
- `run_detection.py` : Helper function for extracting frames from video.
- `requirements.txt` : Required Python packages.

## Notes

- The app processes uploaded videos temporarily and does not store them on the server.
- Analysis time may increase for large videos.
- Only detects camera movement; attempts to filter out object motion as much as possible.

## License

MIT

---

Contributions and feedback are welcome! 
