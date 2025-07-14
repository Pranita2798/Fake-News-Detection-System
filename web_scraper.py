#!/usr/bin/env python3
"""
Web Scraper for Fake News Detection System
Scrapes news articles from various sources for training data collection.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
from urllib.parse import urljoin, urlparse
import csv
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NewsArticleScraper:
    def __init__(self, delay=1):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def scrape_article(self, url):
        """Scrape a single article from the given URL."""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract article data
            article_data = {
                'url': url,
                'title': self._extract_title(soup),
                'content': self._extract_content(soup),
                'author': self._extract_author(soup),
                'publish_date': self._extract_date(soup),
                'source': urlparse(url).netloc,
                'scraped_at': datetime.now().isoformat()
            }
            
            return article_data
            
        except requests.RequestException as e:
            logger.error(f"Error scraping {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error scraping {url}: {e}")
            return None
    
    def _extract_title(self, soup):
        """Extract article title from HTML."""
        selectors = [
            'h1',
            '.entry-title',
            '.post-title',
            '.article-title',
            'title'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)
        
        return "No title found"
    
    def _extract_content(self, soup):
        """Extract article content from HTML."""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        selectors = [
            '.entry-content',
            '.post-content',
            '.article-content',
            '.content',
            'article p',
            '.story-body p'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                content = ' '.join([elem.get_text(strip=True) for elem in elements])
                if len(content) > 100:  # Ensure substantial content
                    return content
        
        # Fallback: extract all paragraphs
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text(strip=True) for p in paragraphs])
        
        return content if content else "No content found"
    
    def _extract_author(self, soup):
        """Extract author information from HTML."""
        selectors = [
            '.author',
            '.byline',
            '.post-author',
            '.article-author',
            '[rel="author"]'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)
        
        return "Unknown"
    
    def _extract_date(self, soup):
        """Extract publication date from HTML."""
        selectors = [
            'time',
            '.publish-date',
            '.post-date',
            '.article-date',
            '.date'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                # Try to get datetime attribute first
                date_str = element.get('datetime') or element.get_text(strip=True)
                return date_str
        
        return "Unknown"
    
    def scrape_multiple_articles(self, urls, output_file='scraped_articles.csv'):
        """Scrape multiple articles and save to CSV."""
        articles = []
        
        for i, url in enumerate(urls, 1):
            logger.info(f"Scraping article {i}/{len(urls)}: {url}")
            
            article_data = self.scrape_article(url)
            if article_data:
                articles.append(article_data)
            
            # Rate limiting
            time.sleep(self.delay)
        
        # Save to CSV
        if articles:
            df = pd.DataFrame(articles)
            df.to_csv(output_file, index=False)
            logger.info(f"Saved {len(articles)} articles to {output_file}")
        
        return articles

def main():
    """Main function to demonstrate web scraping."""
    # Example URLs (replace with actual news sources)
    sample_urls = [
        "https://example-news-site.com/article1",
        "https://example-news-site.com/article2",
        # Add more URLs here
    ]
    
    scraper = NewsArticleScraper(delay=2)
    
    # Scrape articles
    articles = scraper.scrape_multiple_articles(sample_urls)
    
    # Display results
    print(f"Successfully scraped {len(articles)} articles")
    for article in articles[:3]:  # Show first 3 articles
        print(f"\nTitle: {article['title'][:100]}...")
        print(f"Source: {article['source']}")
        print(f"Content length: {len(article['content'])} characters")

if __name__ == "__main__":
    main()