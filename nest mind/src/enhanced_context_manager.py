"""
Enhanced Context Manager - Day 2 implementation with intelligent features
"""
# pylint: disable=import-error,relative-beyond-top-level
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict

from src.nest_mind.core.context_types import (
    ContextItem, ContextPriority, ContextScope, ContextMergeStrategy
)
from src.context_relevance import ContextRelevanceEngine
from src.context_summarizer import ContextSummarizer
from src.nest_mind.core.chat_node import ChatNode
from src.nest_mind.utils.logger import logger

# Try to import persistence, make it optional
try:
    from .nest_mind.context_persistence import ContextPersistence
    PERSISTENCE_AVAILABLE = True
except ImportError:
    PERSISTENCE_AVAILABLE = False
    logger.warning("Context persistence not available")


class EnhancedContextManager:
    """
    Advanced context management system with inheritance, relevance filtering,
    summarization, and persistence
    """
    
    def __init__(self, 
                 max_context_items: int = 50,
                 auto_summarize_threshold: int = 100,
                 enable_persistence: bool = True):
        """
        Initialize enhanced context manager
        
        Args:
            max_context_items: Maximum context items per node
            auto_summarize_threshold: Auto-summarize when items exceed this
            enable_persistence: Enable database persistence
        """
        self.max_context_items = max_context_items
        self.auto_summarize_threshold = auto_summarize_threshold
        self.enable_persistence = enable_persistence and PERSISTENCE_AVAILABLE
        
        # Core engines
        self.relevance_engine = ContextRelevanceEngine()
        self.summarizer = ContextSummarizer()
        
        # Initialize persistence if available and enabled
        if self.enable_persistence:
            try:
                self.persistence = ContextPersistence()
            except Exception as e:
                logger.warning(f"Could not initialize persistence: {e}")
                self.persistence = None
                self.enable_persistence = False
        else:
            self.persistence = None
        
        # Global context storage
        self.global_context: Dict[str, ContextItem] = {}
        
        # Node-specific context cache
        self._context_cache: Dict[str, Dict[str, ContextItem]] = defaultdict(dict)
        
        # Context inheritance rules
        self.inheritance_rules = {
            ContextScope.GLOBAL: True,      # Always inherit global
            ContextScope.INHERITED: True,   # Always inherit from parent
            ContextScope.SHARED: False,     # Only among siblings
            ContextScope.LOCAL: False       # Never inherit
        }
        
        logger.info(f"Initialized EnhancedContextManager (persistence={self.enable_persistence})")
    
    def set_context_item(self, 
                        node_id: str, 
                        key: str, 
                        value: Any,
                        priority: ContextPriority = ContextPriority.MEDIUM,
                        scope: ContextScope = ContextScope.LOCAL,
                        keywords: List[str] = None,
                        expires_in: timedelta = None) -> bool:
        """
        Set a context item for a node
        
        Args:
            node_id: Target node ID
            key: Context key
            value: Context value
            priority: Priority level
            scope: Context scope
            keywords: Keywords for relevance scoring
            expires_in: Time until expiration
            
        Returns:
            True if successful
        """
        try:
            # Create context item
            item = ContextItem(
                key=key,
                value=value,
                priority=priority,
                scope=scope,
                source_node_id=node_id,
                keywords=set(keywords) if keywords else set()
            )
            
            # Set expiration if specified
            if expires_in:
                item.expires_at = datetime.now() + expires_in
            
            # Handle global context
            if scope == ContextScope.GLOBAL:
                self.global_context[key] = item
            else:
                # Store in node-specific context
                self._context_cache[node_id][key] = item
            
            # Persist if enabled
            if self.persistence and scope != ContextScope.GLOBAL:
                node_items = list(self._context_cache[node_id].values())
                self.persistence.save_context_items(node_id, node_items)
            
            # Check if auto-summarization is needed
            total_items = len(self._context_cache[node_id])
            if total_items > self.auto_summarize_threshold:
                self._auto_summarize_context(node_id)
            
            logger.debug(f"Set context item: {node_id}.{key} ({priority.name}, {scope.name})")
            return True
            
        except Exception as e:
            logger.error(f"Error setting context item: {e}")
            return False
    
    def get_context_item(self, 
                        node_id: str, 
                        key: str, 
                        include_inherited: bool = True) -> Optional[ContextItem]:
        """
        Get a specific context item
        
        Args:
            node_id: Source node ID
            key: Context key
            include_inherited: Whether to check inherited context
            
        Returns:
            Context item or None
        """
        # Check local context first
        if key in self._context_cache[node_id]:
            item = self._context_cache[node_id][key]
            if not item.is_expired():
                return item
        
        # Check global context
        if key in self.global_context:
            item = self.global_context[key]
            if not item.is_expired():
                return item
        
        # Check inherited context if requested
        if include_inherited:
            inherited = self.get_inherited_context(node_id)
            if key in inherited:
                item = inherited[key]
                if not item.is_expired():
                    return item
        
        return None
    
    def get_merged_context(self, 
                          node: ChatNode,
                          parent_node: ChatNode = None,
                          conversation_messages: List[str] = None,
                          merge_strategy: ContextMergeStrategy = ContextMergeStrategy.PRIORITY_BASED) -> Dict[str, ContextItem]:
        """
        Get merged context for a node including inherited context
        
        Args:
            node: Current node
            parent_node: Parent node for inheritance
            conversation_messages: Recent messages for relevance filtering
            merge_strategy: Strategy for merging contexts
            
        Returns:
            Merged context dictionary
        """
        try:
            # Start with empty context
            merged_context = {}
            
            # Step 1: Add global context
            for key, item in self.global_context.items():
                if not item.is_expired():
                    merged_context[key] = item
            
            # Step 2: Add inherited context
            if parent_node:
                inherited = self.get_inherited_context(node.id, parent_node.id)
                for key, item in inherited.items():
                    if not item.is_expired():
                        merged_context[key] = self._merge_context_items(
                            merged_context.get(key), item, merge_strategy
                        )
            
            # Step 3: Add local context
            local_context = self._get_local_context(node.id)
            for key, item in local_context.items():
                if not item.is_expired():
                    merged_context[key] = self._merge_context_items(
                        merged_context.get(key), item, merge_strategy
                    )
            
            # Step 4: Apply relevance filtering if conversation provided
            if conversation_messages:
                context_items = list(merged_context.values())
                filtered_items = self.relevance_engine.filter_context_by_relevance(
                    context_items, conversation_messages, self.max_context_items
                )
                merged_context = {item.key: item for item in filtered_items}
            
            logger.debug(f"Merged context for node {node.id}: {len(merged_context)} items")
            return merged_context
            
        except Exception as e:
            logger.error(f"Error merging context: {e}")
            return {}
    
    def get_inherited_context(self, 
                             node_id: str, 
                             parent_id: str = None) -> Dict[str, ContextItem]:
        """
        Get context that should be inherited by a node
        
        Args:
            node_id: Child node ID
            parent_id: Parent node ID
            
        Returns:
            Dictionary of inheritable context items
        """
        inherited_context = {}
        
        if not parent_id:
            return inherited_context
        
        # Get parent's context
        parent_context = self._get_local_context(parent_id)
        
        for key, item in parent_context.items():
            # Check inheritance rules
            if (self.inheritance_rules.get(item.scope, False) and 
                not item.is_expired()):
                
                # Create inherited copy
                inherited_item = ContextItem(
                    key=item.key,
                    value=item.value,
                    priority=item.priority,
                    scope=ContextScope.INHERITED,
                    created_at=item.created_at,
                    updated_at=item.updated_at,
                    source_node_id=item.source_node_id,
                    relevance_score=item.relevance_score,
                    keywords=item.keywords.copy(),
                    metadata=item.metadata.copy(),
                    expires_at=item.expires_at
                )
                
                inherited_context[key] = inherited_item
        
        return inherited_context
    
    def _get_local_context(self, node_id: str) -> Dict[str, ContextItem]:
        """Get local context for a node"""
        # Try cache first
        if node_id in self._context_cache:
            return self._context_cache[node_id].copy()
        
        # Load from persistence if available
        if self.persistence:
            try:
                items = self.persistence.load_context_items(node_id)
                context_dict = {item.key: item for item in items}
                self._context_cache[node_id] = context_dict
                return context_dict.copy()
            except Exception as e:
                logger.warning(f"Could not load context from persistence: {e}")
        
        return {}
    
    def _merge_context_items(self, 
                            existing: Optional[ContextItem], 
                            new: ContextItem,
                            strategy: ContextMergeStrategy) -> ContextItem:
        """Merge two context items based on strategy"""
        if not existing:
            return new
        
        if strategy == ContextMergeStrategy.PRIORITY_BASED:
            return new if new.priority.value >= existing.priority.value else existing
        
        elif strategy == ContextMergeStrategy.TIMESTAMP_BASED:
            return new if new.updated_at >= existing.updated_at else existing
        
        elif strategy == ContextMergeStrategy.WEIGHTED:
            return new if new.relevance_score >= existing.relevance_score else existing
        
        elif strategy == ContextMergeStrategy.CONSERVATIVE:
            # Keep existing, update metadata
            existing.metadata.update(new.metadata)
            existing.keywords.update(new.keywords)
            return existing
        
        elif strategy == ContextMergeStrategy.AGGRESSIVE:
            return new  # Always prefer new
        
        return new  # Default behavior
    
    def _auto_summarize_context(self, node_id: str):
        """Automatically summarize context when it gets too large"""
        try:
            context_items = list(self._context_cache[node_id].values())
            
            # Get recent messages for relevance scoring
            # Note: In a real implementation, you'd pass actual messages
            conversation_messages = []  # Placeholder
            
            # Summarize context
            summarized_items, summary = self.summarizer.summarize_context(
                context_items, conversation_messages, self.max_context_items
            )
            
            # Update cache
            self._context_cache[node_id] = {item.key: item for item in summarized_items}
            
            # Persist changes
            if self.persistence:
                try:
                    self.persistence.save_context_items(node_id, summarized_items)
                    self.persistence.save_context_summary(node_id, summary)
                except Exception as e:
                    logger.warning(f"Could not persist summarized context: {e}")
            
            logger.info(f"Auto-summarized context for node {node_id}: {summary.compression_ratio:.2%} compression")
            
        except Exception as e:
            logger.error(f"Error in auto-summarization: {e}")
    
    def cleanup_expired_context(self):
        """Clean up expired context items"""
        cleaned_count = 0
        
        # Clean global context
        expired_global = [k for k, v in self.global_context.items() if v.is_expired()]
        for key in expired_global:
            del self.global_context[key]
            cleaned_count += 1
        
        # Clean node context
        for node_id in list(self._context_cache.keys()):
            expired_keys = [k for k, v in self._context_cache[node_id].items() if v.is_expired()]
            for key in expired_keys:
                del self._context_cache[node_id][key]
                cleaned_count += 1
        
        # Clean persistence
        if self.persistence:
            try:
                cleaned_count += self.persistence.cleanup_expired_context()
            except Exception as e:
                logger.warning(f"Could not cleanup persistence: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} expired context items")
    
    def get_context_statistics(self) -> Dict[str, Any]:
        """Get comprehensive context statistics"""
        total_global = len(self.global_context)
        total_local = sum(len(context) for context in self._context_cache.values())
        total_nodes = len(self._context_cache)
        
        # Priority distribution
        priority_dist = defaultdict(int)
        for item in self.global_context.values():
            priority_dist[item.priority.name] += 1
        
        for context in self._context_cache.values():
            for item in context.values():
                priority_dist[item.priority.name] += 1
        
        # Database stats
        db_stats = {}
        if self.persistence:
            try:
                db_stats = self.persistence.get_database_stats()
            except Exception as e:
                logger.warning(f"Could not get database stats: {e}")
                db_stats = {"error": str(e)}
        
        return {
            "global_context_items": total_global,
            "local_context_items": total_local,
            "total_context_items": total_global + total_local,
            "nodes_with_context": total_nodes,
            "priority_distribution": dict(priority_dist),
            "database_stats": db_stats,
            "auto_summarize_threshold": self.auto_summarize_threshold,
            "max_items_per_node": self.max_context_items,
            "persistence_enabled": self.enable_persistence
        }
    
    def update_context_item(self, 
                           node_id: str, 
                           key: str, 
                           new_value: Any = None,
                           new_priority: ContextPriority = None,
                           add_keywords: List[str] = None) -> bool:
        """
        Update an existing context item
        
        Args:
            node_id: Node ID
            key: Context key
            new_value: New value (optional)
            new_priority: New priority (optional)  
            add_keywords: Keywords to add (optional)
            
        Returns:
            True if successful
        """
        item = self.get_context_item(node_id, key, include_inherited=False)
        if not item:
            return False
        
        try:
            if new_value is not None:
                item.update_value(new_value)
            
            if new_priority is not None:
                item.priority = new_priority
                item.updated_at = datetime.now()
            
            if add_keywords:
                item.add_keywords(add_keywords)
            
            # Update in cache
            if item.scope == ContextScope.GLOBAL:
                self.global_context[key] = item
            else:
                self._context_cache[node_id][key] = item
            
            # Persist changes
            if self.persistence and item.scope != ContextScope.GLOBAL:
                node_items = list(self._context_cache[node_id].values())
                self.persistence.save_context_items(node_id, node_items)
            
            logger.debug(f"Updated context item: {node_id}.{key}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating context item: {e}")
            return False
    
    def remove_context_item(self, node_id: str, key: str) -> bool:
        """
        Remove a context item
        
        Args:
            node_id: Node ID
            key: Context key
            
        Returns:
            True if successful
        """
        try:
            # Check global context
            if key in self.global_context:
                del self.global_context[key]
                return True
            
            # Check local context
            if key in self._context_cache[node_id]:
                del self._context_cache[node_id][key]
                
                # Persist changes
                if self.persistence:
                    node_items = list(self._context_cache[node_id].values())
                    self.persistence.save_context_items(node_id, node_items)
                
                logger.debug(f"Removed context item: {node_id}.{key}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error removing context item: {e}")
            return False
    
    def get_context_digest(self, node_id: str) -> Dict[str, Any]:
        """
        Get a digestible summary of context for a node
        
        Args:
            node_id: Node ID to get digest for
            
        Returns:
            Dictionary with context digest information
        """
        try:
            local_context = self._get_local_context(node_id)
            global_count = len(self.global_context)
            
            # Use summarizer to create digest
            context_items = list(local_context.values())
            digest = self.summarizer.create_context_digest(context_items)
            
            # Add global context info
            digest["global_context_items"] = global_count
            digest["local_context_items"] = len(context_items)
            
            return digest
            
        except Exception as e:
            logger.error(f"Error creating context digest: {e}")
            return {"error": str(e)}
    
    def export_context(self, node_id: str = None) -> Dict[str, Any]:
        """
        Export context data for backup or transfer
        
        Args:
            node_id: Specific node ID to export, or None for all
            
        Returns:
            Dictionary with exportable context data
        """
        try:
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "global_context": {},
                "node_contexts": {}
            }
            
            # Export global context
            for key, item in self.global_context.items():
                if not item.is_expired():
                    export_data["global_context"][key] = item.to_dict()
            
            # Export node contexts
            if node_id:
                # Export specific node
                local_context = self._get_local_context(node_id)
                export_data["node_contexts"][node_id] = {}
                for key, item in local_context.items():
                    if not item.is_expired():
                        export_data["node_contexts"][node_id][key] = item.to_dict()
            else:
                # Export all nodes
                for nid in self._context_cache:
                    local_context = self._get_local_context(nid)
                    export_data["node_contexts"][nid] = {}
                    for key, item in local_context.items():
                        if not item.is_expired():
                            export_data["node_contexts"][nid][key] = item.to_dict()
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting context: {e}")
            return {"error": str(e)}
    
    def import_context(self, import_data: Dict[str, Any]) -> bool:
        """
        Import context data from backup or transfer
        
        Args:
            import_data: Dictionary with context data to import
            
        Returns:
            True if successful
        """
        try:
            imported_count = 0
            
            # Import global context
            if "global_context" in import_data:
                for key, item_data in import_data["global_context"].items():
                    try:
                        item = ContextItem.from_dict(item_data)
                        if not item.is_expired():
                            self.global_context[key] = item
                            imported_count += 1
                    except Exception as e:
                        logger.warning(f"Could not import global context item {key}: {e}")
            
            # Import node contexts
            if "node_contexts" in import_data:
                for node_id, context_data in import_data["node_contexts"].items():
                    for key, item_data in context_data.items():
                        try:
                            item = ContextItem.from_dict(item_data)
                            if not item.is_expired():
                                self._context_cache[node_id][key] = item
                                imported_count += 1
                        except Exception as e:
                            logger.warning(f"Could not import context item {node_id}.{key}: {e}")
            
            logger.info(f"Imported {imported_count} context items")
            return True
            
        except Exception as e:
            logger.error(f"Error importing context: {e}")
            return False