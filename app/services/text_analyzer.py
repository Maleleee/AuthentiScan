from typing import Dict, Any, List
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch
import numpy as np
from app.core.config import settings
import time
import re
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextAnalyzer:
    def __init__(self):
        try:
            # Initialize zero-shot classification pipeline
            self.zero_shot = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("Zero-shot classification pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing zero-shot pipeline: {str(e)}")
            self.zero_shot = None

        try:
            # Initialize text classification pipeline
            self.text_classifier = pipeline(
                "text-classification",
                model="microsoft/deberta-v3-base",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("Text classification pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing text classifier: {str(e)}")
            self.text_classifier = None

        try:
            # Initialize GPT-2 for perplexity calculation
            self.gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
            logger.info("GPT-2 model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing GPT-2 model: {str(e)}")
            self.gpt2_tokenizer = None
            self.gpt2_model = None

    def calculate_perplexity(self, text: str) -> float:
        """Calculate text perplexity using GPT-2"""
        try:
            if not self.gpt2_tokenizer or not self.gpt2_model:
                logger.warning("GPT-2 model not initialized, skipping perplexity calculation")
                return 0.0

            inputs = self.gpt2_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.gpt2_model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
            return perplexity
        except Exception as e:
            logger.error(f"Error calculating perplexity: {str(e)}")
            return 0.0

    def analyze_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Analyze linguistic features that might indicate AI generation"""
        try:
            # Split text into sentences
            sentences = re.split(r'[.!?]+', text)
            
            # Calculate various metrics
            avg_sentence_length = np.mean([len(s.split()) for s in sentences])
            sentence_length_variance = np.var([len(s.split()) for s in sentences])
            
            # Calculate word frequency distribution
            words = text.lower().split()
            word_freq = Counter(words)
            unique_words = len(word_freq)
            total_words = len(words)
            vocabulary_richness = unique_words / total_words
            
            # Calculate punctuation patterns
            punctuation_count = len(re.findall(r'[.,!?;:]', text))
            punctuation_density = punctuation_count / total_words
            
            # Calculate confidence based on metrics
            confidence = 0.0
            if vocabulary_richness > 0.7:  # High vocabulary richness
                confidence += 0.3
            if punctuation_density > 0.1:  # Natural punctuation
                confidence += 0.2
            if 5 < avg_sentence_length < 25:  # Natural sentence length
                confidence += 0.3
            if sentence_length_variance > 5:  # Natural variation in sentence length
                confidence += 0.2
            
            return {
                "avg_sentence_length": avg_sentence_length,
                "sentence_length_variance": sentence_length_variance,
                "vocabulary_richness": vocabulary_richness,
                "punctuation_density": punctuation_density,
                "confidence": min(confidence, 1.0)
            }
        except Exception as e:
            logger.error(f"Error in linguistic analysis: {str(e)}")
            return {"confidence": 0.0}

    def analyze_with_zero_shot(self, text: str) -> Dict[str, Any]:
        """Analyze text using zero-shot classification"""
        try:
            if not self.zero_shot:
                logger.warning("Zero-shot model not initialized, skipping zero-shot analysis")
                return {'confidence': 0.0}

            result = self.zero_shot(
                text,
                candidate_labels=["human-written", "AI-generated"],
                multi_label=False
            )
            return {
                'confidence': result['scores'][0],
                'label': result['labels'][0]
            }
        except Exception as e:
            logger.error(f"Error in zero-shot analysis: {str(e)}")
            return {'confidence': 0.0}

    def analyze_with_text_classifier(self, text: str) -> Dict[str, Any]:
        """Analyze text using text classification"""
        try:
            if not self.text_classifier:
                logger.warning("Text classifier not initialized, skipping classification analysis")
                return {'confidence': 0.0}

            result = self.text_classifier(text)
            return {
                'confidence': result[0]['score'],
                'label': result[0]['label']
            }
        except Exception as e:
            logger.error(f"Error in text classification: {str(e)}")
            return {'confidence': 0.0}

    async def analyze_text(self, text: str, source: str = "unknown") -> Dict[str, Any]:
        """Main method to analyze text content"""
        try:
            logger.info(f"Starting text analysis for source: {source}")
            
            # Initialize results dictionary
            results = {
                'source': source,
                'timestamp': time.time(),
                'analysis_methods': {}
            }

            # Calculate perplexity
            perplexity = self.calculate_perplexity(text)
            results['analysis_methods']['perplexity'] = {
                'score': perplexity,
                'confidence': 1.0 - min(perplexity / 100, 1.0)  # Convert perplexity to confidence
            }
            logger.info(f"Perplexity calculation completed: {perplexity}")

            # Linguistic analysis
            linguistic_result = self.analyze_linguistic_features(text)
            results['analysis_methods']['linguistic'] = linguistic_result
            logger.info(f"Linguistic analysis completed: {linguistic_result}")

            # Zero-shot classification
            zero_shot_result = self.analyze_with_zero_shot(text)
            results['analysis_methods']['zero_shot'] = zero_shot_result
            logger.info(f"Zero-shot analysis completed: {zero_shot_result}")

            # Text classification
            classifier_result = self.analyze_with_text_classifier(text)
            results['analysis_methods']['text_classifier'] = classifier_result
            logger.info(f"Text classification completed: {classifier_result}")

            # Calculate overall confidence score
            confidences = []
            for method, result in results['analysis_methods'].items():
                if isinstance(result, dict) and 'confidence' in result:
                    confidences.append(result['confidence'])
            
            if confidences:
                overall_confidence = sum(confidences) / len(confidences)
            else:
                overall_confidence = 0.0

            # Add confidence score and AI generation flag
            results['confidence_score'] = overall_confidence
            results['is_ai_generated'] = overall_confidence > 0.7

            logger.info(f"Text analysis completed successfully with overall confidence: {overall_confidence}")
            return results

        except Exception as e:
            logger.error(f"Error in text analysis: {str(e)}")
            raise Exception(f"Failed to analyze text: {str(e)}")

# Create a singleton instance
text_analyzer = TextAnalyzer() 