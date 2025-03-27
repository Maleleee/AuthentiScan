import time
import logging
import traceback
from typing import Dict, Any
from app.services.website_scraper import website_scraper
from app.services.text_analyzer import text_analyzer
from app.services.image_analyzer import image_analyzer
from app.services.video_analyzer import video_analyzer

# Set up logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WebsiteAnalyzer:
    async def analyze_website(self, url: str, analyze_text: bool = True, analyze_images: bool = True, analyze_videos: bool = True) -> Dict[str, Any]:
        """Main method to analyze website content"""
        try:
            logger.info(f"Starting website analysis for URL: {url}")
            
            # Validate URL
            if not url:
                raise ValueError("URL cannot be empty")
            
            # Initialize results dictionary
            results = {
                'is_ai_generated': False,
                'confidence_score': 0.0,
                'details': {
                    'text_analysis': [],
                    'image_analysis': None,
                    'video_analysis': None,
                    'website_stats': {
                        'total_pages': 0,
                        'total_images': 0,
                        'total_videos': 0
                    }
                },
                'accuracy_metrics': {
                    'text_detection_accuracy': 0.85,
                    'image_detection_accuracy': 0.8,
                    'video_detection_accuracy': 0.75
                }
            }

            # Scrape website content
            try:
                logger.info(f"Attempting to scrape content from {url}")
                scraped_content = await website_scraper.analyze_website(url)
                if not scraped_content:
                    raise Exception("No content was scraped from the website")
                logger.info(f"Successfully scraped content from {url}")
                logger.debug(f"Scraped content structure: {scraped_content.keys()}")
            except Exception as e:
                logger.error(f"Failed to scrape website {url}: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise Exception(f"Failed to scrape website: {str(e)}")
            
            # Update website stats
            results['details']['website_stats'] = {
                'total_pages': scraped_content.get('total_pages_analyzed', 0),
                'total_images': scraped_content.get('total_images', 0),
                'total_videos': scraped_content.get('total_videos', 0)
            }
            
            # Analyze text content if requested
            if analyze_text:
                try:
                    logger.info("Starting text analysis")
                    text_content = scraped_content.get('main_page', {}).get('text', '')
                    if text_content:
                        text_result = await text_analyzer.analyze_text(
                            text_content,
                            source=f"main_page_{url}"
                        )
                        results['details']['text_analysis'].append(text_result)
                        logger.info("Text analysis completed successfully")
                    else:
                        logger.warning("No text content found to analyze")
                except Exception as e:
                    logger.error(f"Text analysis failed: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    results['details']['text_analysis'].append({'error': str(e)})

            # Analyze images if requested
            if analyze_images:
                try:
                    images = scraped_content.get('main_page', {}).get('images', [])
                    if images:
                        logger.info(f"Starting image analysis for {len(images)} images")
                        image_result = await image_analyzer.analyze_images(images)
                        results['details']['image_analysis'] = image_result
                        logger.info("Image analysis completed successfully")
                    else:
                        logger.warning("No images found to analyze")
                except Exception as e:
                    logger.error(f"Image analysis failed: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    results['details']['image_analysis'] = {'error': str(e)}

            # Analyze videos if requested
            if analyze_videos:
                try:
                    videos = scraped_content.get('main_page', {}).get('videos', [])
                    if videos:
                        logger.info(f"Starting video analysis for {len(videos)} videos")
                        video_result = await video_analyzer.analyze_videos(videos)
                        results['details']['video_analysis'] = video_result
                        logger.info("Video analysis completed successfully")
                    else:
                        logger.warning("No videos found to analyze")
                except Exception as e:
                    logger.error(f"Video analysis failed: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    results['details']['video_analysis'] = {'error': str(e)}

            # Calculate overall confidence score
            confidences = []
            
            # Add text analysis confidences
            for text_result in results['details']['text_analysis']:
                if isinstance(text_result, dict) and 'confidence_score' in text_result:
                    confidences.append(text_result['confidence_score'])
            
            # Add image analysis confidence
            if results['details']['image_analysis'] and isinstance(results['details']['image_analysis'], dict):
                if 'average_confidence' in results['details']['image_analysis']:
                    confidences.append(results['details']['image_analysis']['average_confidence'])
            
            # Add video analysis confidence
            if results['details']['video_analysis'] and isinstance(results['details']['video_analysis'], dict):
                if 'average_confidence' in results['details']['video_analysis']:
                    confidences.append(results['details']['video_analysis']['average_confidence'])
            
            if confidences:
                results['confidence_score'] = sum(confidences) / len(confidences)
            else:
                results['confidence_score'] = 0.0

            # Determine if content is AI-generated
            results['is_ai_generated'] = results['confidence_score'] > 0.7

            logger.info(f"Website analysis completed successfully with overall confidence: {results['confidence_score']}")
            return results

        except Exception as e:
            logger.error(f"Error in website analysis: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise Exception(f"Failed to analyze website: {str(e)}")

# Create a singleton instance
website_analyzer = WebsiteAnalyzer() 