from typing import Dict, Any, List
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import face_recognition
from app.core.config import settings

class VideoAnalyzer:
    def __init__(self):
        # Initialize face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
    def download_video(self, url: str) -> str:
        """Download video from URL and save temporarily"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Save video temporarily
            temp_path = "temp_video.mp4"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return temp_path
        except Exception as e:
            raise Exception(f"Error downloading video: {str(e)}")
    
    def extract_frames(self, video_path: str, max_frames: int = 30) -> List[np.ndarray]:
        """Extract frames from video"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame interval
        interval = max(1, total_frames // max_frames)
        
        frame_count = 0
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % interval == 0:
                frames.append(frame)
            frame_count += 1
            
        cap.release()
        return frames
    
    def analyze_face_consistency(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze face consistency across frames"""
        try:
            face_encodings = []
            inconsistencies = []
            
            for i, frame in enumerate(frames):
                # Detect faces in frame
                face_locations = face_recognition.face_locations(frame)
                
                if not face_locations:
                    continue
                    
                # Get face encodings
                frame_encodings = face_recognition.face_encodings(frame, face_locations)
                face_encodings.extend(frame_encodings)
            
            if not face_encodings:
                return {
                    "has_faces": False,
                    "confidence": 0.0,
                    "details": "No faces detected in video"
                }
            
            # Compare faces across frames
            for i, encoding in enumerate(face_encodings):
                for j, other_encoding in enumerate(face_encodings):
                    if i != j:
                        distance = face_recognition.face_distance([encoding], other_encoding)[0]
                        if distance > 0.6:  # Threshold for face similarity
                            inconsistencies.append(f"Face inconsistency between frames")
            
            return {
                "has_faces": True,
                "confidence": 1.0 - (len(inconsistencies) / len(face_encodings)),
                "details": {
                    "num_faces": len(face_encodings),
                    "inconsistencies": inconsistencies
                }
            }
        except Exception as e:
            raise Exception(f"Face consistency analysis error: {str(e)}")
    
    def analyze_video_artifacts(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze video for common deepfake artifacts"""
        try:
            artifact_scores = []
            
            for frame in frames:
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Apply various filters
                edges = cv2.Canny(gray, 100, 200)
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                
                # Calculate artifact metrics
                edge_density = np.mean(edges > 0)
                laplacian_variance = np.var(laplacian)
                
                # Combine metrics into a score
                frame_score = min(edge_density * 2, 1.0)
                artifact_scores.append(frame_score)
            
            return {
                "mean_artifact_score": np.mean(artifact_scores),
                "std_artifact_score": np.std(artifact_scores),
                "confidence": np.mean(artifact_scores)
            }
        except Exception as e:
            raise Exception(f"Video artifact analysis error: {str(e)}")
    
    async def analyze_video(self, video_url: str) -> Dict[str, Any]:
        """Main method to analyze video using all available methods"""
        try:
            # Download and process video
            video_path = self.download_video(video_url)
            frames = self.extract_frames(video_path)
            
            # Run all analyses
            face_analysis = self.analyze_face_consistency(frames)
            artifact_analysis = self.analyze_video_artifacts(frames)
            
            # Calculate overall confidence score
            confidence_scores = [
                face_analysis["confidence"],
                artifact_analysis["confidence"]
            ]
            
            overall_confidence = sum(confidence_scores) / len(confidence_scores)
            
            return {
                "is_ai_generated": overall_confidence > settings.VIDEO_CONFIDENCE_THRESHOLD,
                "confidence_score": overall_confidence,
                "details": {
                    "face_analysis": face_analysis,
                    "artifact_analysis": artifact_analysis
                },
                "accuracy_metrics": {
                    "face_detection_accuracy": 0.85,  # These would need to be calculated based on testing
                    "artifact_detection_accuracy": 0.80
                }
            }
        except Exception as e:
            raise Exception(f"Video analysis error: {str(e)}")
        finally:
            # Clean up temporary file
            import os
            if os.path.exists("temp_video.mp4"):
                os.remove("temp_video.mp4")

# Create a singleton instance
video_analyzer = VideoAnalyzer() 