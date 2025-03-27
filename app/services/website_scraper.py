from typing import Dict, Any, List
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import aiohttp
import asyncio
from app.core.config import settings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebsiteScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.timeout = 30  # 30 seconds timeout
    
    async def fetch_url(self, session: aiohttp.ClientSession, url: str) -> str:
        """Fetch URL content asynchronously"""
        try:
            async with session.get(url, headers=self.headers, timeout=self.timeout) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    logger.warning(f"Failed to fetch {url}: {response.status}")
                    return ""
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return ""
    
    def extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract text content from BeautifulSoup object"""
        try:
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            logger.error(f"Error extracting text content: {str(e)}")
            return ""
    
    def extract_images(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract image URLs from BeautifulSoup object"""
        try:
            images = []
            for img in soup.find_all('img'):
                src = img.get('src')
                if src:
                    # Convert relative URLs to absolute URLs
                    absolute_url = urljoin(base_url, src)
                    if absolute_url.startswith(('http://', 'https://')):
                        images.append(absolute_url)
            return images
        except Exception as e:
            logger.error(f"Error extracting images: {str(e)}")
            return []
    
    def extract_videos(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract video URLs from BeautifulSoup object"""
        try:
            videos = []
            
            # Check video tags
            for video in soup.find_all('video'):
                src = video.get('src')
                if src:
                    absolute_url = urljoin(base_url, src)
                    if absolute_url.startswith(('http://', 'https://')):
                        videos.append(absolute_url)
            
            # Check source tags within video elements
            for source in soup.find_all('source'):
                src = source.get('src')
                if src:
                    absolute_url = urljoin(base_url, src)
                    if absolute_url.startswith(('http://', 'https://')):
                        videos.append(absolute_url)
            
            return videos
        except Exception as e:
            logger.error(f"Error extracting videos: {str(e)}")
            return []
    
    def extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract links from BeautifulSoup object"""
        try:
            links = []
            base_domain = urlparse(base_url).netloc
            
            for a in soup.find_all('a', href=True):
                href = a['href']
                # Convert relative URLs to absolute URLs
                absolute_url = urljoin(base_url, href)
                # Only include links from the same domain
                if urlparse(absolute_url).netloc == base_domain:
                    links.append(absolute_url)
            return links
        except Exception as e:
            logger.error(f"Error extracting links: {str(e)}")
            return []
    
    async def analyze_website(self, url: str) -> Dict[str, Any]:
        """Main method to analyze website content"""
        try:
            # Validate URL
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            async with aiohttp.ClientSession() as session:
                # Fetch main page
                html_content = await self.fetch_url(session, url)
                if not html_content:
                    raise Exception(f"Failed to fetch content from {url}")
                
                soup = BeautifulSoup(html_content, 'lxml')
                
                # Extract content
                text_content = self.extract_text_content(soup)
                images = self.extract_images(soup, url)
                videos = self.extract_videos(soup, url)
                links = self.extract_links(soup, url)
                
                # Fetch additional pages (up to 5 internal pages)
                additional_pages = []
                for link in links[:5]:
                    try:
                        page_content = await self.fetch_url(session, link)
                        if page_content:
                            page_soup = BeautifulSoup(page_content, 'lxml')
                            additional_pages.append({
                                'url': link,
                                'text': self.extract_text_content(page_soup),
                                'images': self.extract_images(page_soup, link),
                                'videos': self.extract_videos(page_soup, link)
                            })
                    except Exception as e:
                        logger.warning(f"Error fetching additional page {link}: {str(e)}")
                
                return {
                    'main_page': {
                        'url': url,
                        'text': text_content,
                        'images': images,
                        'videos': videos
                    },
                    'additional_pages': additional_pages,
                    'total_pages_analyzed': len(additional_pages) + 1,
                    'total_images': len(images) + sum(len(page['images']) for page in additional_pages),
                    'total_videos': len(videos) + sum(len(page['videos']) for page in additional_pages)
                }
        except Exception as e:
            logger.error(f"Website analysis error: {str(e)}")
            raise Exception(f"Failed to analyze website: {str(e)}")

# Create a singleton instance
website_scraper = WebsiteScraper() 