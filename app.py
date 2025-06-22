import streamlit as st
import os
import tempfile
import cv2

from movement_detector import CameraMovementDetector 
from run_detection import extract_frames_from_video

def main():
    st.set_page_config(page_title="Camera Movement Detector", layout="wide")
    
    st.title("📷 Advanced Camera Movement Detector")
    st.write(
        "Upload a video to detect camera movements. "
        "The algorithm uses feature matching and homography decomposition to distinguish between "
        "translation, rotation, and scaling. It also tries to differentiate between "
        "true camera motion and object motion within the scene."
    )
    
    # --- Sidebar for settings ---
    st.sidebar.header("⚙️ Detection Thresholds")
    
    translation_threshold = st.sidebar.slider(
        "Translation Threshold (% of diagonal)",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.1,
        help="Detects pans/tilts. Percentage of the frame's diagonal."
    )
    rotation_threshold = st.sidebar.slider(
        "Rotation Threshold (degrees)",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="Detects rotation."
    )
    scale_threshold = st.sidebar.slider(
        "Scale Threshold (percent)",
        min_value=1.0,
        max_value=50.0,
        value=5.0,
        step=0.5,
        help="Detects zoom. "
    )
    
    st.sidebar.header("⚙️ Fine-Tuning")
    inlier_ratio_thresh = st.sidebar.slider(
        "Inlier Ratio Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Minimum ratio of matching points that fit the camera model. Higher values reject object motion."
    )
    orb_features = st.sidebar.slider(
        "Feature Count",
        min_value=500,
        max_value=5000,
        value=2000,
        step=100,
        help="Number of features to detect in each frame."
    )

    st.sidebar.header("⚙️ General Settings")
    num_frames_to_sample = st.sidebar.slider(
        "Number of Frames to Sample",
        min_value=10,
        max_value=50,
        value=20,
        step=1,
        help="How many frames to extract from the video for analysis. More frames are more accurate but slower."
    )

    # --- Main content ---
    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_video_path = tmp_file.name
        
        st.video(temp_video_path)
        
        if st.button("Detect Movement"):
            with st.spinner("Analyzing video... This may take a moment."):
                frames = extract_frames_from_video(temp_video_path, num_frames_to_sample)
                
                if not frames:
                    st.error("Could not process the video. Please try a different file.")
                else:
                    # Yeni sınıfı başlat ve kullan
                    detector = CameraMovementDetector(
                        translation_thresh_px=translation_threshold / 100.0, # Yüzdeyi orana çevir
                        rotation_thresh_deg=rotation_threshold,
                        scale_thresh=scale_threshold / 100.0, # Yüzdeyi orana çevir
                        inlier_ratio_thresh=inlier_ratio_thresh,
                        orb_features=orb_features
                    )
                    movement_details = detector.detect(frames)
                    
                    st.success("Analysis complete!")
                    
                    if not movement_details:
                        st.info("✅ No significant camera movement was detected based on the current settings.")
                    else:
                        num_movements = len(movement_details)
                        st.warning(f"🚨 Found {num_movements} instances of significant camera movement.")
                        
                        st.subheader("Detected Movement Details:")
                        
                        for movement in movement_details:
                            idx = movement['frame_index']
                            reasons = ", ".join(movement['reasons']).title()
                            
                            col1, col2 = st.columns([1, 3])
                            
                            with col1:
                                if idx < len(frames):
                                    st.image(cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB), caption=f"Frame {idx}", use_container_width=True)
                            
                            with col2:
                                st.markdown(f"**Movement at Frame {idx} → {idx+1}**")
                                st.markdown(f"**Type:** {reasons}")
                                st.markdown(
                                    f"- **Translation:** `{movement['translation_px']:.2f}` pixels "
                                    f"- **Rotation:** `{movement['rotation_deg']:.2f}`° "
                                    f"- **Scale Change:** `{movement['scale_diff']*100:.2f}`%"
                                )
                                st.markdown(f"- **Inlier Ratio:** `{movement['inlier_ratio']:.2f}`")

                            st.divider()

            # Clean up the temporary file
            os.remove(temp_video_path)

if __name__ == "__main__":
    main() 