"""
Production-ready YAKE keyword extraction service with fallback search capabilities.
Supports intelligent document indexing, context retrieval, and robust error handling.
"""

import logging
import traceback
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import math

try:
    import yake
    YAKE_AVAILABLE = True
except ImportError:
    YAKE_AVAILABLE = False
    logging.warning("YAKE not available. Keyword extraction will use fallback methods.")

from app.core.config import settings
from app.services.parser_service import DocumentChunk, ParsedDocument

logger = logging.getLogger(__name__)


@dataclass
class ExtractedKeyword:
    """Represents an extracted keyword with its score and metadata."""
    keyword: str
    score: float
    frequency: int
    positions: List[int]
    chunks: List[int]  # Chunk indices where keyword appears
    context_snippets: List[str]


@dataclass
class SearchResult:
    """Represents a search result with relevance scoring."""
    chunk_index: int
    chunk: DocumentChunk
    relevance_score: float
    matched_keywords: List[str]
    context_snippet: str
    highlight_positions: List[Tuple[int, int]]


@dataclass
class DocumentIndex:
    """Document index with keywords and search capabilities."""
    keywords: List[ExtractedKeyword]
    keyword_map: Dict[str, ExtractedKeyword]
    chunk_keywords: Dict[int, List[str]]  # chunk_index -> keywords
    total_chunks: int
    extraction_method: str
    creation_stats: Dict[str, Any]


class YAKEExtractionError(Exception):
    """Raised when YAKE extraction fails."""
    pass


class YAKEService:
    """
    Production-ready YAKE keyword extraction service.
    
    Features:
    - YAKE-based keyword extraction with configurable parameters
    - Fallback to TF-IDF and frequency-based methods when YAKE fails
    - Intelligent search with relevance scoring
    - Context-aware result ranking
    - Memory-efficient indexing for large documents
    - Comprehensive error handling and graceful degradation
    """
    
    def __init__(self):
        self.max_keywords = getattr(settings, 'YAKE_MAX_KEYWORDS', 20)
        self.ngram_size = getattr(settings, 'YAKE_NGRAM_SIZE', 3)
        self.enable_yake = getattr(settings, 'ENABLE_YAKE_SEARCH', True)
        
        # YAKE configuration
        self.yake_params = {
            'lan': 'en',  # Language
            'n': self.ngram_size,  # N-gram size
            'dedupLim': 0.7,  # Deduplication threshold
            'windowsSize': 1,  # Window size for co-occurrence
            'top': self.max_keywords  # Number of keywords to extract
        }
        
        # Validate YAKE availability
        if self.enable_yake and not YAKE_AVAILABLE:
            logger.warning("YAKE enabled but not available. Falling back to frequency-based extraction.")
            self.enable_yake = False
    
    async def create_document_index(self, parsed_document: ParsedDocument) -> DocumentIndex:
        """
        Create searchable index from parsed document.
        
        Args:
            parsed_document: Parsed document with chunks and content
            
        Returns:
            DocumentIndex with keywords and search capabilities
            
        Raises:
            YAKEExtractionError: If keyword extraction fails completely
        """
        
        try:
            logger.info(f"Creating document index - {len(parsed_document.chunks)} chunks")
            
            start_time = logger.handlers[0].formatter.formatTime if logger.handlers else None
            
            # Extract keywords using primary method
            if self.enable_yake and YAKE_AVAILABLE:
                try:
                    keywords = await self._extract_keywords_yake(parsed_document)
                    extraction_method = "YAKE"
                    logger.info(f"YAKE extraction successful - {len(keywords)} keywords")
                except Exception as e:
                    logger.warning(f"YAKE extraction failed: {e}. Falling back to frequency-based method.")
                    keywords = await self._extract_keywords_fallback(parsed_document)
                    extraction_method = "Frequency-based (YAKE fallback)"
            else:
                keywords = await self._extract_keywords_fallback(parsed_document)
                extraction_method = "Frequency-based (YAKE disabled)"
            
            # Create keyword map for efficient lookups
            keyword_map = {kw.keyword.lower(): kw for kw in keywords}
            
            # Create chunk-to-keywords mapping
            chunk_keywords = defaultdict(list)
            for keyword in keywords:
                for chunk_idx in keyword.chunks:
                    chunk_keywords[chunk_idx].append(keyword.keyword)
            
            # Calculate creation statistics
            creation_stats = {
                "total_keywords": len(keywords),
                "extraction_method": extraction_method,
                "chunks_with_keywords": len(chunk_keywords),
                "average_keywords_per_chunk": sum(len(kws) for kws in chunk_keywords.values()) / len(chunk_keywords) if chunk_keywords else 0,
                "top_keywords": [kw.keyword for kw in keywords[:10]],
                "yake_available": YAKE_AVAILABLE,
                "yake_enabled": self.enable_yake
            }
            
            logger.info(
                f"Document index created successfully - "
                f"{len(keywords)} keywords, "
                f"{len(chunk_keywords)} indexed chunks, "
                f"method: {extraction_method}"
            )
            
            return DocumentIndex(
                keywords=keywords,
                keyword_map=keyword_map,
                chunk_keywords=dict(chunk_keywords),
                total_chunks=len(parsed_document.chunks),
                extraction_method=extraction_method,
                creation_stats=creation_stats
            )
            
        except Exception as e:
            error_msg = f"Failed to create document index: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise YAKEExtractionError(error_msg) from e
    
    async def _extract_keywords_yake(self, parsed_document: ParsedDocument) -> List[ExtractedKeyword]:
        """Extract keywords using YAKE algorithm."""
        
        # Create YAKE extractor
        kw_extractor = yake.KeywordExtractor(**self.yake_params)
        
        # Extract keywords from full document text
        yake_keywords = kw_extractor.extract_keywords(parsed_document.content)
        
        # Convert to our keyword format with additional metadata
        keywords = []
        for score, keyword_text in yake_keywords:
            # Find keyword occurrences in chunks
            positions, chunks, snippets = self._find_keyword_occurrences(
                keyword_text, 
                parsed_document
            )
            
            if positions:  # Only include keywords that appear in the text
                keywords.append(ExtractedKeyword(
                    keyword=keyword_text,
                    score=score,
                    frequency=len(positions),
                    positions=positions,
                    chunks=chunks,
                    context_snippets=snippets
                ))
        
        # Sort by YAKE score (lower is better for YAKE)
        keywords.sort(key=lambda x: x.score)
        
        return keywords[:self.max_keywords]
    
    async def _extract_keywords_fallback(self, parsed_document: ParsedDocument) -> List[ExtractedKeyword]:
        """Fallback keyword extraction using frequency and TF-IDF-like scoring."""
        
        # Tokenize and clean text
        text = parsed_document.content.lower()
        
        # Remove common stop words and punctuation
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'among', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you',
            'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my',
            'your', 'his', 'her', 'its', 'our', 'their'
        }
        
        # Extract candidate terms (words and phrases)
        candidates = self._extract_candidate_terms(text, stop_words)
        
        # Calculate scores for candidates
        keyword_scores = {}
        for candidate in candidates:
            # Calculate frequency-based score
            frequency = candidates[candidate]
            
            # Calculate inverse document frequency (chunk-based)
            chunks_with_term = sum(
                1 for chunk in parsed_document.chunks 
                if candidate.lower() in chunk.content.lower()
            )
            
            # Simple TF-IDF-like score
            tf = frequency / len(text.split())
            idf = math.log(len(parsed_document.chunks) / max(1, chunks_with_term))
            score = tf * idf
            
            keyword_scores[candidate] = score
        
        # Convert to ExtractedKeyword objects
        keywords = []
        for candidate, score in sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True):
            positions, chunks, snippets = self._find_keyword_occurrences(
                candidate, 
                parsed_document
            )
            
            if positions:
                keywords.append(ExtractedKeyword(
                    keyword=candidate,
                    score=score,
                    frequency=len(positions),
                    positions=positions,
                    chunks=chunks,
                    context_snippets=snippets
                ))
        
        return keywords[:self.max_keywords]
    
    def _extract_candidate_terms(self, text: str, stop_words: Set[str]) -> Dict[str, int]:
        """Extract candidate terms from text with frequency counts."""
        
        candidates = defaultdict(int)
        
        # Clean text and split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            # Clean sentence
            sentence = re.sub(r'[^\w\s]', ' ', sentence.lower())
            words = sentence.split()
            
            # Filter words
            filtered_words = [
                word for word in words 
                if len(word) > 2 and word not in stop_words and word.isalpha()
            ]
            
            # Extract single words
            for word in filtered_words:
                candidates[word] += 1
            
            # Extract phrases (2-3 words)
            for i in range(len(filtered_words) - 1):
                # 2-word phrases
                phrase = ' '.join(filtered_words[i:i+2])
                if len(phrase) > 5:  # Minimum phrase length
                    candidates[phrase] += 1
                
                # 3-word phrases
                if i < len(filtered_words) - 2:
                    phrase = ' '.join(filtered_words[i:i+3])
                    if len(phrase) > 8:  # Minimum phrase length
                        candidates[phrase] += 1
        
        # Filter out very rare terms (appear less than 2 times)
        return {term: count for term, count in candidates.items() if count >= 2}
    
    def _find_keyword_occurrences(
        self, 
        keyword: str, 
        parsed_document: ParsedDocument
    ) -> Tuple[List[int], List[int], List[str]]:
        """Find all occurrences of a keyword in the document."""
        
        positions = []
        chunks = []
        snippets = []
        
        # Search in full content for positions
        text = parsed_document.content.lower()
        keyword_lower = keyword.lower()
        
        start = 0
        while True:
            pos = text.find(keyword_lower, start)
            if pos == -1:
                break
            positions.append(pos)
            
            # Extract context snippet
            snippet_start = max(0, pos - 50)
            snippet_end = min(len(text), pos + len(keyword) + 50)
            snippet = parsed_document.content[snippet_start:snippet_end]
            snippets.append(snippet.strip())
            
            start = pos + 1
        
        # Find chunks containing the keyword
        chunk_indices = []
        for i, chunk in enumerate(parsed_document.chunks):
            if keyword_lower in chunk.content.lower():
                chunk_indices.append(i)
        
        return positions, chunk_indices, snippets
    
    async def search_document(
        self, 
        query: str, 
        document_index: DocumentIndex, 
        parsed_document: ParsedDocument,
        max_results: int = 5
    ) -> List[SearchResult]:
        """
        Search document using keyword index and return relevant chunks.
        
        Args:
            query: Search query
            document_index: Pre-built document index
            parsed_document: Original parsed document
            max_results: Maximum number of results to return
            
        Returns:
            List of SearchResult objects sorted by relevance
        """
        
        try:
            logger.info(f"Searching document for query: '{query[:100]}...'")
            
            # Tokenize query and extract keywords
            query_terms = self._extract_query_terms(query)
            
            # Score chunks based on keyword matches
            chunk_scores = defaultdict(float)
            chunk_matches = defaultdict(list)
            
            for term in query_terms:
                # Find matching keywords in index
                matching_keywords = self._find_matching_keywords(term, document_index)
                
                for keyword_obj in matching_keywords:
                    # Add relevance score for each chunk containing this keyword
                    for chunk_idx in keyword_obj.chunks:
                        # Calculate relevance score (lower YAKE score = higher relevance)
                        if document_index.extraction_method.startswith("YAKE"):
                            relevance = 1.0 / (keyword_obj.score + 0.1)  # Avoid division by zero
                        else:
                            relevance = keyword_obj.score
                        
                        # Boost score based on frequency and exact matches
                        frequency_boost = math.log(keyword_obj.frequency + 1)
                        exact_match_boost = 2.0 if term.lower() == keyword_obj.keyword.lower() else 1.0
                        
                        chunk_scores[chunk_idx] += relevance * frequency_boost * exact_match_boost
                        chunk_matches[chunk_idx].append(keyword_obj.keyword)
            
            # Fallback: if no keyword matches, use simple text search
            if not chunk_scores:
                logger.info("No keyword matches found, using fallback text search")
                chunk_scores, chunk_matches = self._fallback_text_search(
                    query_terms, parsed_document
                )
            
            # Create search results
            search_results = []
            for chunk_idx in sorted(chunk_scores.keys(), key=lambda x: chunk_scores[x], reverse=True):
                if len(search_results) >= max_results:
                    break
                
                chunk = parsed_document.chunks[chunk_idx]
                
                # Create context snippet with highlights
                context_snippet, highlight_positions = self._create_context_snippet(
                    chunk.content, 
                    chunk_matches[chunk_idx],
                    max_length=200
                )
                
                search_results.append(SearchResult(
                    chunk_index=chunk_idx,
                    chunk=chunk,
                    relevance_score=chunk_scores[chunk_idx],
                    matched_keywords=list(set(chunk_matches[chunk_idx])),
                    context_snippet=context_snippet,
                    highlight_positions=highlight_positions
                ))
            
            logger.info(f"Search completed - {len(search_results)} results found")
            
            return search_results
            
        except Exception as e:
            error_msg = f"Document search failed: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            # Return empty results rather than failing completely
            return []
    
    def _extract_query_terms(self, query: str) -> List[str]:
        """Extract searchable terms from query."""
        
        # Clean query
        query = re.sub(r'[^\w\s]', ' ', query.lower())
        words = query.split()
        
        # Filter out very short words and common terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'is', 'are', 'what', 'how', 'where', 'when', 'why'}
        
        terms = [word for word in words if len(word) > 2 and word not in stop_words]
        
        # Add phrases
        for i in range(len(terms) - 1):
            phrase = ' '.join(terms[i:i+2])
            terms.append(phrase)
        
        return terms
    
    def _find_matching_keywords(self, term: str, document_index: DocumentIndex) -> List[ExtractedKeyword]:
        """Find keywords that match a search term."""
        
        matching_keywords = []
        term_lower = term.lower()
        
        for keyword in document_index.keywords:
            keyword_lower = keyword.keyword.lower()
            
            # Exact match
            if term_lower == keyword_lower:
                matching_keywords.append(keyword)
            # Partial match
            elif term_lower in keyword_lower or keyword_lower in term_lower:
                matching_keywords.append(keyword)
        
        return matching_keywords
    
    def _fallback_text_search(
        self, 
        query_terms: List[str], 
        parsed_document: ParsedDocument
    ) -> Tuple[Dict[int, float], Dict[int, List[str]]]:
        """Fallback text search when keyword matching fails."""
        
        chunk_scores = defaultdict(float)
        chunk_matches = defaultdict(list)
        
        for chunk_idx, chunk in enumerate(parsed_document.chunks):
            content_lower = chunk.content.lower()
            
            for term in query_terms:
                term_lower = term.lower()
                
                # Count occurrences
                count = content_lower.count(term_lower)
                if count > 0:
                    # Simple scoring based on frequency and term length
                    score = count * len(term) / len(chunk.content)
                    chunk_scores[chunk_idx] += score
                    chunk_matches[chunk_idx].append(term)
        
        return dict(chunk_scores), dict(chunk_matches)
    
    def _create_context_snippet(
        self, 
        text: str, 
        matched_keywords: List[str], 
        max_length: int = 200
    ) -> Tuple[str, List[Tuple[int, int]]]:
        """Create context snippet with highlight positions."""
        
        if len(text) <= max_length:
            return text, []
        
        # Find best snippet location based on keyword density
        best_start = 0
        best_score = 0
        
        # Try different starting positions
        for start in range(0, len(text) - max_length + 1, 20):
            snippet = text[start:start + max_length]
            
            # Score based on keyword matches
            score = 0
            for keyword in matched_keywords:
                score += snippet.lower().count(keyword.lower())
            
            if score > best_score:
                best_score = score
                best_start = start
        
        # Extract best snippet
        snippet = text[best_start:best_start + max_length]
        
        # Find highlight positions within snippet
        highlight_positions = []
        for keyword in matched_keywords:
            start_pos = 0
            keyword_lower = keyword.lower()
            snippet_lower = snippet.lower()
            
            while True:
                pos = snippet_lower.find(keyword_lower, start_pos)
                if pos == -1:
                    break
                highlight_positions.append((pos, pos + len(keyword)))
                start_pos = pos + 1
        
        return snippet, highlight_positions
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get current service statistics and configuration."""
        return {
            "yake_available": YAKE_AVAILABLE,
            "yake_enabled": self.enable_yake,
            "max_keywords": self.max_keywords,
            "ngram_size": self.ngram_size,
            "yake_params": self.yake_params if YAKE_AVAILABLE else None,
            "fallback_methods": ["Frequency-based", "TF-IDF-like", "Simple text search"]
        }


# Global YAKE service instance
yake_service = YAKEService()