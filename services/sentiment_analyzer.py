from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    def __init__(self):
        """Initialize BERT model for sentiment analysis"""
        try:
            # Use BERT model fine-tuned for sentiment analysis
            model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

            logger.info("Loading BERT sentiment model...")
            # Load model and tokenizer
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Create pipeline for easier inference
            self.analyzer = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1,  # Use CPU, change to 0 for GPU
            )
            logger.info("BERT sentiment model loaded successfully")

        except Exception as e:
            logger.error(f"Error initializing BERT model: {e}")
            self.analyzer = None

    def analyze_batch(self, texts, weights=None):
        """Analyze multiple texts with weighted scoring"""
        if not texts:
            return 0.0

        try:
            # Default weights give more importance to recent content
            if weights is None:
                weights = [1.0 - (i * 0.8 / len(texts)) for i in range(len(texts))]

            scores = []
            for text in texts:
                # Truncate text to BERT's maximum length
                if len(text) > self.tokenizer.model_max_length:
                    text = text[: self.tokenizer.model_max_length]

                # Tokenize and get sentiment
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
                with torch.no_grad():
                    outputs = self.model(**inputs)

                # Get predicted sentiment (1-5 scale)
                predicted_class = outputs.logits.argmax().item() + 1

                # Convert to -1 to 1 scale
                score = (predicted_class - 3) / 2
                scores.append(score)

            # Calculate weighted average
            weighted_score = np.average(scores, weights=weights)

            # Apply sigmoid-like normalization
            final_score = 2 / (1 + np.exp(-weighted_score)) - 1

            logger.info(
                f"Analyzed {len(texts)} texts. Scores: {scores}, Final: {final_score:.2f}"
            )
            return final_score

        except Exception as e:
            logger.error(f"Error in BERT sentiment analysis: {e}")
            return 0.0

    def _get_sentiment_explanation(self, score):
        """Get human-readable explanation of sentiment score"""
        if score > 0.6:
            return "Very Positive"
        elif score > 0.2:
            return "Positive"
        elif score < -0.6:
            return "Very Negative"
        elif score < -0.2:
            return "Negative"
        else:
            return "Neutral"
