from typing import Dict, Any, List
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import face_recognition
from app.core.config import settings
import logging
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImageAnalyzer:
    def __init__(self):
        # Load the StyleGAN detector model
        # This would need to be implemented with the actual model
        self.stylegan_detector = None
        
    def download_image(self, url: str) -> Image.Image:
        """Download image from URL and convert to PIL Image"""
        try:
            logger.info(f"Attempting to download image from {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            logger.info(f"Successfully downloaded image from {url}")
            return image
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download image from {url}: {str(e)}")
            raise Exception(f"Failed to download image: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error downloading image from {url}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise Exception(f"Error processing image: {str(e)}")
    
    def analyze_face_consistency(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze face consistency using face_recognition library"""
        try:
            logger.info("Starting face consistency analysis")
            # Convert PIL Image to numpy array
            img_array = np.array(image)
            
            # Detect faces
            face_locations = face_recognition.face_locations(img_array)
            
            if not face_locations:
                logger.info("No faces detected in image")
                return {
                    "has_face": False,
                    "confidence": 0.0,
                    "details": "No face detected"
                }
            
            logger.info(f"Found {len(face_locations)} faces in image")
            
            # Analyze face consistency
            face_encodings = face_recognition.face_encodings(img_array, face_locations)
            
            # Check for inconsistencies in face features
            inconsistencies = []
            for i, encoding in enumerate(face_encodings):
                # Compare with other faces in the image
                for j, other_encoding in enumerate(face_encodings):
                    if i != j:
                        distance = face_recognition.face_distance([encoding], other_encoding)[0]
                        if distance > 0.6:  # Threshold for face similarity
                            inconsistencies.append(f"Face {i} and {j} are significantly different")
            
            confidence = 1.0 - (len(inconsistencies) / len(face_encodings)) if face_encodings else 0.0
            logger.info(f"Face analysis completed with confidence: {confidence}")
            
            return {
                "has_face": True,
                "confidence": confidence,
                "details": {
                    "num_faces": len(face_locations),
                    "inconsistencies": inconsistencies
                }
            }
        except Exception as e:
            logger.error(f"Face analysis error: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "has_face": False,
                "confidence": 0.0,
                "details": f"Error in face analysis: {str(e)}"
            }
    
    def analyze_image_artifacts(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image for common AI generation artifacts"""
        try:
            logger.info("Starting artifact analysis")
            # Convert to numpy array for OpenCV
            img_array = np.array(image)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply various filters to detect artifacts
            edges = cv2.Canny(gray, 100, 200)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            
            # Calculate artifact metrics
            edge_density = np.mean(edges > 0)
            laplacian_variance = np.var(laplacian)
            
            confidence = min(edge_density * 2, 1.0)  # Simple confidence metric
            logger.info(f"Artifact analysis completed with confidence: {confidence}")
            
            return {
                "edge_density": edge_density,
                "laplacian_variance": laplacian_variance,
                "confidence": confidence
            }
        except Exception as e:
            logger.error(f"Artifact analysis error: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "edge_density": 0.0,
                "laplacian_variance": 0.0,
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def analyze_image(self, image_url: str) -> Dict[str, Any]:
        """Analyze a single image"""
        try:
            logger.info(f"Starting analysis for image: {image_url}")
            
            # Download image
            image = self.download_image(image_url)
            
            # Run all analyses
            face_analysis = self.analyze_face_consistency(image)
            artifact_analysis = self.analyze_image_artifacts(image)
            
            # Calculate overall confidence score
            confidence_scores = [
                face_analysis["confidence"],
                artifact_analysis["confidence"]
            ]
            
            overall_confidence = sum(confidence_scores) / len(confidence_scores)
            logger.info(f"Image analysis completed with overall confidence: {overall_confidence}")
            
            return {
                "url": image_url,
                "is_ai_generated": overall_confidence > settings.IMAGE_CONFIDENCE_THRESHOLD,
                "confidence_score": overall_confidence,
                "details": {
                    "face_analysis": face_analysis,
                    "artifact_analysis": artifact_analysis
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing image {image_url}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "url": image_url,
                "error": str(e),
                "is_ai_generated": False,
                "confidence_score": 0.0
            }

    async def analyze_images(self, image_urls: List[str]) -> Dict[str, Any]:
        """Analyze multiple images"""
        try:
            logger.info(f"Starting batch analysis for {len(image_urls)} images")
            results = []
            
            for url in image_urls:
                try:
                    result = await self.analyze_image(url)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to analyze image {url}: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    results.append({
                        "url": url,
                        "error": str(e),
                        "is_ai_generated": False,
                        "confidence_score": 0.0
                    })

            # Calculate overall statistics
            valid_results = [r for r in results if "error" not in r]
            if valid_results:
                avg_confidence = sum(r["confidence_score"] for r in valid_results) / len(valid_results)
                ai_generated_count = sum(1 for r in valid_results if r["is_ai_generated"])
            else:
                avg_confidence = 0.0
                ai_generated_count = 0

            logger.info(f"Batch analysis completed. Successfully analyzed {len(valid_results)}/{len(image_urls)} images")
            return {
                "total_images": len(image_urls),
                "successfully_analyzed": len(valid_results),
                "ai_generated_count": ai_generated_count,
                "average_confidence": avg_confidence,
                "results": results
            }
        except Exception as e:
            logger.error(f"Error in batch image analysis: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise Exception(f"Failed to analyze images: {str(e)}")

# Create a singleton instance
image_analyzer = ImageAnalyzer() 