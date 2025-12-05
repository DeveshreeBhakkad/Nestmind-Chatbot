"""
Context summarization engine for compressing context while preserving key information
"""
from typing import List, Dict, Any, Set, Tuple
from collections import Counter, defaultdict
import json
from datetime import datetime

from src.nest_mind.core.context_types import ContextItem, ContextPriority, ContextSummary
from src.context_relevance import ContextRelevanceEngine
from src.nest_mind.utils.logger import logger


class ContextSummarizer:
    """
    Engine for summarizing context to reduce information overload
    """
    
    def __init__(self, 
                 max_summary_items: int = 20,
                 compression_ratio: float = 0.3,
                 preserve_critical: bool = True):
        """
        Initialize context summarizer
        
        Args:
            max_summary_items: Maximum items to keep in summary
            compression_ratio: Target compression ratio (0.3 = keep 30%)
            preserve_critical: Always preserve critical priority items
        """
        self.max_summary_items = max_summary_items
        self.compression_ratio = compression_ratio
        self.preserve_critical = preserve_critical
        self.relevance_engine = ContextRelevanceEngine()
        
        logger.info("Initialized ContextSummarizer")
    
    def summarize_context(self, 
                         context_items: List[ContextItem],
                         conversation_messages: List[str] = None,
                         target_size: int = None) -> Tuple[List[ContextItem], ContextSummary]:
        """
        Summarize a list of context items
        
        Args:
            context_items: Items to summarize
            conversation_messages: Recent conversation for relevance scoring
            target_size: Target number of items (overrides compression_ratio)
            
        Returns:
            Tuple of (summarized_items, summary_metadata)
        """
        if not context_items:
            return [], ContextSummary(0, 0, 1.0, [], [])
        
        original_count = len(context_items)
        
        # Calculate target size
        if target_size is None:
            target_size = max(1, int(original_count * self.compression_ratio))
        target_size = min(target_size, self.max_summary_items)
        
        logger.info(f"Summarizing {original_count} context items to {target_size}")
        
        # Step 1: Remove expired items
        valid_items = [item for item in context_items if not item.is_expired()]
        
        # Step 2: Always preserve critical items
        critical_items = []
        other_items = []
        
        for item in valid_items:
            if self.preserve_critical and item.priority == ContextPriority.CRITICAL:
                critical_items.append(item)
            else:
                other_items.append(item)
        
        # Step 3: Score relevance if conversation provided
        if conversation_messages:
            for item in other_items:
                self.relevance_engine.score_context_relevance(
                    item, conversation_messages
                )
        
        # Step 4: Apply summarization strategies
        remaining_slots = target_size - len(critical_items)
        if remaining_slots > 0 and other_items:
            summarized_other = self._apply_summarization_strategies(
                other_items, remaining_slots, conversation_messages
            )
        else:
            summarized_other = []
        
        # Step 5: Combine results
        final_items = critical_items + summarized_other
        
        # Step 6: Create summary metadata
        key_points = self._extract_key_points(final_items)
        preserved_priorities = list(set(item.priority for item in final_items))
        
        summary = ContextSummary(
            original_items=original_count,
            summarized_items=len(final_items),
            compression_ratio=len(final_items) / original_count if original_count > 0 else 1.0,
            key_points=key_points,
            preserved_priorities=preserved_priorities
        )
        
        logger.info(
            f"Context summarization complete: {original_count} -> {len(final_items)} "
            f"(ratio: {summary.compression_ratio:.2%})"
        )
        
        return final_items, summary
    
    def _apply_summarization_strategies(self, 
                                      items: List[ContextItem], 
                                      target_count: int,
                                      conversation_messages: List[str] = None) -> List[ContextItem]:
        """Apply multiple summarization strategies"""
        
        if len(items) <= target_count:
            return items
        
        # Strategy 1: Priority-based filtering
        priority_filtered = self._filter_by_priority(items, target_count * 2)
        
        # Strategy 2: Relevance-based filtering (if conversation available)
        if conversation_messages:
            relevance_filtered = self._filter_by_relevance(
                priority_filtered, conversation_messages, target_count
            )
        else:
            relevance_filtered = priority_filtered[:target_count]
        
        # Strategy 3: Temporal filtering (prefer recent items)
        final_items = self._filter_by_recency(relevance_filtered, target_count)
        
        return final_items
    
    def _filter_by_priority(self, items: List[ContextItem], target_count: int) -> List[ContextItem]:
        """Filter items by priority level"""
        # Sort by priority (high to low)
        priority_order = [
            ContextPriority.HIGH,
            ContextPriority.MEDIUM,
            ContextPriority.LOW,
            ContextPriority.MINIMAL
        ]
        
        sorted_items = sorted(
            items, 
            key=lambda x: priority_order.index(x.priority) if x.priority in priority_order else 999
        )
        
        return sorted_items[:target_count]
    
    def _filter_by_relevance(self, 
                           items: List[ContextItem], 
                           conversation_messages: List[str], 
                           target_count: int) -> List[ContextItem]:
        """Filter items by relevance to conversation"""
        # Items should already have relevance scores from earlier scoring
        sorted_items = sorted(items, key=lambda x: x.relevance_score, reverse=True)
        return sorted_items[:target_count]
    
    def _filter_by_recency(self, items: List[ContextItem], target_count: int) -> List[ContextItem]:
        """Filter items preferring more recent ones"""
        sorted_items = sorted(items, key=lambda x: x.updated_at, reverse=True)
        return sorted_items[:target_count]
    
    def _extract_key_points(self, items: List[ContextItem]) -> List[str]:
        """Extract key points from summarized context"""
        key_points = []
        
        # Group by priority
        priority_groups = defaultdict(list)
        for item in items:
            priority_groups[item.priority].append(item)
        
        # Extract points by priority
        for priority in [ContextPriority.CRITICAL, ContextPriority.HIGH, ContextPriority.MEDIUM]:
            if priority in priority_groups:
                count = len(priority_groups[priority])
                key_points.append(f"{count} {priority.name.lower()} priority items preserved")
        
        # Extract common keywords
        all_keywords = set()
        for item in items:
            if hasattr(item, 'keywords'):
                all_keywords.update(item.keywords)
        
        if all_keywords:
            top_keywords = list(all_keywords)[:5]  # Top 5 keywords
            key_points.append(f"Key topics: {', '.join(top_keywords)}")
        
        return key_points
    
    def create_context_digest(self, 
                            context_items: List[ContextItem]) -> Dict[str, Any]:
        """
        Create a digestible summary of context for display
        
        Args:
            context_items: Context items to create digest from
            
        Returns:
            Dictionary with digest information
        """
        if not context_items:
            return {"total": 0, "summary": "No context available"}
        
        # Group by priority
        priority_counts = Counter(item.priority for item in context_items)
        
        # Group by scope
        scope_counts = Counter(item.scope for item in context_items)
        
        # Get recent items
        recent_items = sorted(
            context_items, 
            key=lambda x: x.updated_at, 
            reverse=True
        )[:5]
        
        # Calculate age distribution
        now = datetime.now()
        age_distribution = {
            "recent": 0,    # < 1 hour
            "today": 0,     # < 24 hours
            "week": 0,      # < 7 days
            "older": 0      # >= 7 days
        }
        
        for item in context_items:
            age_hours = (now - item.updated_at).total_seconds() / 3600
            if age_hours < 1:
                age_distribution["recent"] += 1
            elif age_hours < 24:
                age_distribution["today"] += 1
            elif age_hours < 168:  # 7 days
                age_distribution["week"] += 1
            else:
                age_distribution["older"] += 1
        
        return {
            "total": len(context_items),
            "priority_distribution": {p.name: count for p, count in priority_counts.items()},
            "scope_distribution": {s.name: count for s, count in scope_counts.items()},
            "age_distribution": age_distribution,
            "recent_keys": [item.key for item in recent_items],
            "avg_relevance": sum(item.relevance_score for item in context_items) / len(context_items),
            "summary": f"{len(context_items)} context items with {priority_counts.get(ContextPriority.CRITICAL, 0)} critical"
        }
    
    def merge_similar_contexts(self, 
                             context_items: List[ContextItem],
                             similarity_threshold: float = 0.8) -> List[ContextItem]:
        """
        Merge similar context items to reduce redundancy
        
        Args:
            context_items: Items to check for similarity
            similarity_threshold: Threshold for considering items similar
            
        Returns:
            List with similar items merged
        """
        if len(context_items) <= 1:
            return context_items
        
        merged_items = []
        processed_indices = set()
        
        for i, item in enumerate(context_items):
            if i in processed_indices:
                continue
            
            # Find similar items
            similar_items = [item]
            for j, other_item in enumerate(context_items[i+1:], i+1):
                if j in processed_indices:
                    continue
                
                # Calculate similarity based on keywords
                if item.keywords and other_item.keywords:
                    similarity = len(item.keywords.intersection(other_item.keywords)) / \
                                len(item.keywords.union(other_item.keywords))
                    
                    if similarity >= similarity_threshold:
                        similar_items.append(other_item)
                        processed_indices.add(j)
            
            # Merge similar items
            if len(similar_items) > 1:
                merged_item = self._merge_context_items(similar_items)
                merged_items.append(merged_item)
            else:
                merged_items.append(item)
            
            processed_indices.add(i)
        
        if len(merged_items) < len(context_items):
            logger.info(f"Merged {len(context_items)} items into {len(merged_items)}")
        
        return merged_items
    
    def _merge_context_items(self, items: List[ContextItem]) -> ContextItem:
        """Merge multiple similar context items into one"""
        if len(items) == 1:
            return items[0]
        
        # Use the highest priority item as base
        base_item = max(items, key=lambda x: x.priority.value)
        
        # Merge values (combine if they're strings, otherwise take the base)
        merged_value = base_item.value
        if isinstance(merged_value, str):
            other_values = [str(item.value) for item in items if item != base_item]
            if other_values:
                merged_value = f"{merged_value}; {'; '.join(other_values)}"
        
        # Merge keywords
        merged_keywords = set()
        for item in items:
            merged_keywords.update(item.keywords)
        
        # Merge metadata
        merged_metadata = {}
        for item in items:
            merged_metadata.update(item.metadata)
        
        # Create merged item
        merged_item = ContextItem(
            key=f"merged_{base_item.key}",
            value=merged_value,
            priority=base_item.priority,
            scope=base_item.scope,
            created_at=min(item.created_at for item in items),
            updated_at=max(item.updated_at for item in items),
            source_node_id=base_item.source_node_id,
            relevance_score=max(item.relevance_score for item in items),
            keywords=merged_keywords,
            metadata=merged_metadata
        )
        
        # Add merge info to metadata
        merged_item.metadata["merged_from"] = [item.key for item in items]
        merged_item.metadata["merge_count"] = len(items)
        
        return merged_item