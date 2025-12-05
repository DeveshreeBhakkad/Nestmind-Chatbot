"""
Context relevance scoring and filtering system
"""
import re
import math
from typing import Dict, List, Set, Tuple, Any
from collections import Counter

try:
    from src.nest_mind.core.context_types import ContextItem, ContextPriority
    from src.nest_mind.utils.logger import logger
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.nest_mind.core.context_types import ContextItem, ContextPriority
    from src.nest_mind.utils.logger import logger

# Try to import advanced NLP libraries, fall back to basic processing if not available
try:
    from textblob import TextBlob
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    ADVANCED_NLP = True
except ImportError:
    ADVANCED_NLP = False
    logger.warning("Advanced NLP libraries not available, using basic text processing")


class ContextRelevanceEngine:
    """
    Engine for scoring context relevance and filtering based on conversation content
    """
    
    def __init__(self, 
                 min_relevance_threshold: float = 0.3,
                 keyword_weight: float = 0.6,
                 semantic_weight: float = 0.4):
        """
        Initialize relevance engine
        
        Args:
            min_relevance_threshold: Minimum relevance score to keep context
            keyword_weight: Weight for keyword-based scoring
            semantic_weight: Weight for semantic similarity scoring
        """
        self.min_relevance_threshold = min_relevance_threshold
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
        
        # Initialize components based on available libraries
        if ADVANCED_NLP:
            self.vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=500,
                ngram_range=(1, 2),
                min_df=1
            )
        else:
            self.vectorizer = None
        
        # Cache for performance
        self._keyword_cache: Dict[str, Set[str]] = {}
        
        logger.info(f"Initialized ContextRelevanceEngine (advanced_nlp={ADVANCED_NLP})")
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> Set[str]:
        """
        Extract keywords from text
        
        Args:
            text: Input text to analyze
            max_keywords: Maximum number of keywords to return
            
        Returns:
            Set of extracted keywords
        """
        if not text or not text.strip():
            return set()
        
        # Check cache first
        cache_key = f"{hash(text)}_{max_keywords}"
        if cache_key in self._keyword_cache:
            return self._keyword_cache[cache_key]
        
        keywords = set()
        
        try:
            if ADVANCED_NLP:
                keywords = self._extract_keywords_advanced(text, max_keywords)
            else:
                keywords = self._extract_keywords_basic(text, max_keywords)
        except Exception as e:
            logger.warning(f"Error extracting keywords: {e}")
            keywords = self._extract_keywords_basic(text, max_keywords)
        
        # Cache the result
        self._keyword_cache[cache_key] = keywords
        return keywords
    
    def _extract_keywords_advanced(self, text: str, max_keywords: int) -> Set[str]:
        """Extract keywords using advanced NLP"""
        keywords = set()
        
        # Clean and normalize text
        cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Use TextBlob for noun phrase extraction
        blob = TextBlob(text)
        noun_phrases = [str(phrase).strip() for phrase in blob.noun_phrases]
        
        # Add meaningful noun phrases
        for phrase in noun_phrases[:max_keywords//2]:
            if len(phrase) > 2:
                keywords.add(phrase)
        
        # Get individual words and their frequencies
        words = cleaned_text.split()
        word_freq = Counter(words)
        
        # Add high-frequency single words
        for word, freq in word_freq.most_common(max_keywords):
            if len(word) > 2 and word.isalpha():
                keywords.add(word)
            
            if len(keywords) >= max_keywords:
                break
        
        return keywords
    
    def _extract_keywords_basic(self, text: str, max_keywords: int) -> Set[str]:
        """Extract keywords using basic text processing"""
        # Simple keyword extraction using word frequency
        words = re.findall(r'\b\w{3,}\b', text.lower())
        
        # Filter out common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
            'her', 'its', 'our', 'their', 'a', 'an'
        }
        
        filtered_words = [word for word in words if word not in stop_words]
        word_freq = Counter(filtered_words)
        
        return set([word for word, freq in word_freq.most_common(max_keywords)])
    
    def calculate_keyword_similarity(self, 
                                   context_keywords: Set[str], 
                                   conversation_keywords: Set[str]) -> float:
        """
        Calculate similarity based on keyword overlap
        
        Args:
            context_keywords: Keywords from context item
            conversation_keywords: Keywords from conversation
            
        Returns:
            Similarity score between 0 and 1
        """
        if not context_keywords or not conversation_keywords:
            return 0.0
        
        # Jaccard similarity (intersection over union)
        intersection = len(context_keywords.intersection(conversation_keywords))
        union = len(context_keywords.union(conversation_keywords))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_semantic_similarity(self, 
                                    context_text: str, 
                                    conversation_text: str) -> float:
        """
        Calculate semantic similarity using TF-IDF and cosine similarity
        
        Args:
            context_text: Text from context
            conversation_text: Text from conversation
            
        Returns:
            Semantic similarity score between 0 and 1
        """
        if not context_text or not conversation_text or not ADVANCED_NLP:
            return 0.0
        
        try:
            # Prepare texts
            texts = [context_text, conversation_text]
            
            # Generate TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Return similarity between the two texts
            return float(similarity_matrix[0, 1])
            
        except Exception as e:
            logger.warning(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def score_context_relevance(self, 
                              context_item: ContextItem, 
                              conversation_messages: List[str],
                              conversation_keywords: Set[str] = None) -> float:
        """
        Score the relevance of a context item to current conversation
        
        Args:
            context_item: Context item to score
            conversation_messages: Recent conversation messages
            conversation_keywords: Pre-extracted conversation keywords
            
        Returns:
            Relevance score between 0 and 1
        """
        if not conversation_messages:
            return 0.0
        
        # Combine conversation messages
        conversation_text = " ".join(conversation_messages)
        
        # Extract keywords if not provided
        if conversation_keywords is None:
            conversation_keywords = self.extract_keywords(conversation_text)
        
        # Extract context text and keywords
        context_text = str(context_item.value)
        if context_item.keywords:
            context_keywords = context_item.keywords
        else:
            context_keywords = self.extract_keywords(context_text)
            # Update the context item with extracted keywords
            context_item.keywords = context_keywords
        
        # Calculate keyword similarity
        keyword_similarity = self.calculate_keyword_similarity(
            context_keywords, conversation_keywords
        )
        
        # Calculate semantic similarity
        semantic_similarity = self.calculate_semantic_similarity(
            context_text, conversation_text
        )
        
        # Combine scores with weights
        relevance_score = (
            self.keyword_weight * keyword_similarity + 
            self.semantic_weight * semantic_similarity
        )
        
        # Apply priority boost
        priority_boost = self._get_priority_boost(context_item.priority)
        relevance_score = min(1.0, relevance_score * priority_boost)
        
        # Update the context item's relevance score
        context_item.relevance_score = relevance_score
        
        logger.debug(
            f"Context relevance - Key: {context_item.key}, "
            f"Keyword: {keyword_similarity:.3f}, "
            f"Semantic: {semantic_similarity:.3f}, "
            f"Final: {relevance_score:.3f}"
        )
        
        return relevance_score
    
    def _get_priority_boost(self, priority: ContextPriority) -> float:
        """Get boost factor based on context priority"""
        boost_map = {
            ContextPriority.CRITICAL: 2.0,
            ContextPriority.HIGH: 1.5,
            ContextPriority.MEDIUM: 1.0,
            ContextPriority.LOW: 0.8,
            ContextPriority.MINIMAL: 0.6
        }
        return boost_map.get(priority, 1.0)
    
    def filter_context_by_relevance(self, 
                                   context_items: List[ContextItem], 
                                   conversation_messages: List[str],
                                   max_items: int = None) -> List[ContextItem]:
        """
        Filter context items based on relevance to conversation
        
        Args:
            context_items: List of context items to filter
            conversation_messages: Recent conversation messages
            max_items: Maximum number of items to return
            
        Returns:
            Filtered list of relevant context items
        """
        if not context_items:
            return []
        
        # Extract conversation keywords once
        conversation_text = " ".join(conversation_messages)
        conversation_keywords = self.extract_keywords(conversation_text)
        
        # Score all context items
        scored_items = []
        for item in context_items:
            if item.is_expired():
                continue  # Skip expired items
            
            score = self.score_context_relevance(
                item, conversation_messages, conversation_keywords
            )
            
            # Keep items above threshold or with high priority
            if (score >= self.min_relevance_threshold or 
                item.priority in [ContextPriority.CRITICAL, ContextPriority.HIGH]):
                scored_items.append((score, item))
        
        # Sort by relevance score (descending)
        scored_items.sort(key=lambda x: x[0], reverse=True)
        
        # Apply max_items limit if specified
        if max_items:
            scored_items = scored_items[:max_items]
        
        # Return just the items
        filtered_items = [item for score, item in scored_items]
        
        logger.info(
            f"Filtered context: {len(context_items)} -> {len(filtered_items)} items "
            f"(threshold: {self.min_relevance_threshold})"
        )
        
        return filtered_items
    
    def get_relevance_stats(self, context_items: List[ContextItem]) -> Dict[str, Any]:
        """Get statistics about context relevance"""
        if not context_items:
            return {}
        
        scores = [item.relevance_score for item in context_items]
        
        return {
            "total_items": len(context_items),
            "avg_relevance": sum(scores) / len(scores) if scores else 0,
            "min_relevance": min(scores) if scores else 0,
            "max_relevance": max(scores) if scores else 0,
            "above_threshold": sum(1 for s in scores if s >= self.min_relevance_threshold),
            "threshold": self.min_relevance_threshold
        }