from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, Any, List
import uvicorn
from app.services.text_analyzer import text_analyzer
from app.services.image_analyzer import image_analyzer
from app.services.video_analyzer import video_analyzer
from app.services.website_scraper import website_scraper
from app.core.config import settings

app = FastAPI(
    title="AuthentiScan API",
    description="API for detecting AI-generated content across various platforms",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class WebsiteAnalysisRequest(BaseModel):
    url: HttpUrl
    analyze_text: bool = True
    analyze_images: bool = True
    analyze_videos: bool = True

class TextAnalysisRequest(BaseModel):
    text: str
    source: Optional[str] = None

class ImageAnalysisRequest(BaseModel):
    image_url: HttpUrl
    source: Optional[str] = None

class VideoAnalysisRequest(BaseModel):
    video_url: HttpUrl
    source: Optional[str] = None

class AnalysisResponse(BaseModel):
    is_ai_generated: bool
    confidence_score: float
    details: Dict[str, Any]
    accuracy_metrics: Dict[str, float]

@app.get("/")
async def root():
    return {"message": "Welcome to AuthentiScan API"}

@app.post("/api/analyze/website", response_model=AnalysisResponse)
async def analyze_website(request: WebsiteAnalysisRequest):
    """
    Analyze a website for AI-generated content
    """
    try:
        # Scrape website content
        website_content = await website_scraper.analyze_website(str(request.url))
        
        # Initialize results
        text_results = []
        image_results = []
        video_results = []
        
        # Analyze main page content
        if request.analyze_text:
            text_results.append(await text_analyzer.analyze_text(website_content['main_page']['text']))
        
        if request.analyze_images:
            for image_url in website_content['main_page']['images']:
                image_results.append(await image_analyzer.analyze_image(image_url))
        
        if request.analyze_videos:
            for video_url in website_content['main_page']['videos']:
                video_results.append(await video_analyzer.analyze_video(video_url))
        
        # Analyze additional pages
        for page in website_content['additional_pages']:
            if request.analyze_text:
                text_results.append(await text_analyzer.analyze_text(page['text']))
            
            if request.analyze_images:
                for image_url in page['images']:
                    image_results.append(await image_analyzer.analyze_image(image_url))
            
            if request.analyze_videos:
                for video_url in page['videos']:
                    video_results.append(await video_analyzer.analyze_video(video_url))
        
        # Calculate overall scores
        text_score = sum(r['confidence_score'] for r in text_results) / len(text_results) if text_results else 0
        image_score = sum(r['confidence_score'] for r in image_results) / len(image_results) if image_results else 0
        video_score = sum(r['confidence_score'] for r in video_results) / len(video_results) if video_results else 0
        
        # Calculate weighted average
        total_weight = sum([
            len(text_results) if request.analyze_text else 0,
            len(image_results) if request.analyze_images else 0,
            len(video_results) if request.analyze_videos else 0
        ])
        
        if total_weight == 0:
            raise HTTPException(status_code=400, detail="No content to analyze")
        
        overall_confidence = (
            (text_score * len(text_results) if request.analyze_text else 0) +
            (image_score * len(image_results) if request.analyze_images else 0) +
            (video_score * len(video_results) if request.analyze_videos else 0)
        ) / total_weight
        
        return {
            "is_ai_generated": overall_confidence > 0.7,  # Threshold for AI detection
            "confidence_score": overall_confidence,
            "details": {
                "text_analysis": text_results if request.analyze_text else None,
                "image_analysis": image_results if request.analyze_images else None,
                "video_analysis": video_results if request.analyze_videos else None,
                "website_stats": {
                    "total_pages": website_content['total_pages_analyzed'],
                    "total_images": website_content['total_images'],
                    "total_videos": website_content['total_videos']
                }
            },
            "accuracy_metrics": {
                "text_detection_accuracy": 0.85,  # These would need to be calculated based on testing
                "image_detection_accuracy": 0.80,
                "video_detection_accuracy": 0.75
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/text", response_model=AnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """
    Analyze text content for AI generation
    """
    try:
        result = await text_analyzer.analyze_text(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/image", response_model=AnalysisResponse)
async def analyze_image(request: ImageAnalysisRequest):
    """
    Analyze image content for AI generation
    """
    try:
        result = await image_analyzer.analyze_image(str(request.image_url))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/video", response_model=AnalysisResponse)
async def analyze_video(request: VideoAnalysisRequest):
    """
    Analyze video content for deepfakes
    """
    try:
        result = await video_analyzer.analyze_video(str(request.video_url))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 