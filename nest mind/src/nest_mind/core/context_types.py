"""
Context types and data structures for advanced context management
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from enum import Enum


class ContextPriority(Enum):
    """Priority levels for context items"""
    CRITICAL = 5    # Must always be preserved
    HIGH = 4       # Very important, rarely filtered
    MEDIUM = 3     # Standard importance
    LOW = 2        # Can be filtered if space is limited
    MINIMAL = 1    # First to be filtered out


class ContextScope(Enum):
    """Scope of context propagation"""
    GLOBAL = "global"        # Available to all nodes
    INHERITED = "inherited"  # Passed down from parents
    LOCAL = "local"         # Only in current node
    SHARED = "shared"       # Shared among siblings


@dataclass
class ContextItem:
    """Individual context item with metadata"""
    key: str
    value: Any
    priority: ContextPriority = ContextPriority.MEDIUM
    scope: ContextScope = ContextScope.LOCAL
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    source_node_id: str = ""
    relevance_score: float = 1.0
    keywords: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    expires_at: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if context item has expired"""
        return self.expires_at is not None and datetime.now() > self.expires_at
    
    def update_value(self, new_value: Any):
        """Update the value and timestamp"""
        self.value = new_value
        self.updated_at = datetime.now()
    
    def add_keywords(self, keywords: List[str]):
        """Add keywords for relevance scoring"""
        self.keywords.update(keywords)
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "key": self.key,
            "value": self.value,
            "priority": self.priority.value,
            "scope": self.scope.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "source_node_id": self.source_node_id,
            "relevance_score": self.relevance_score,
            "keywords": list(self.keywords),
            "metadata": self.metadata,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextItem':
        """Create from dictionary"""
        item = cls(
            key=data["key"],
            value=data["value"],
            priority=ContextPriority(data["priority"]),
            scope=ContextScope(data["scope"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            source_node_id=data.get("source_node_id", ""),
            relevance_score=data.get("relevance_score", 1.0),
            keywords=set(data.get("keywords", [])),
            metadata=data.get("metadata", {})
        )
        
        if data.get("expires_at"):
            item.expires_at = datetime.fromisoformat(data["expires_at"])
        
        return item


@dataclass
class ContextSummary:
    """Summary of context information"""
    original_items: int
    summarized_items: int
    compression_ratio: float
    key_points: List[str]
    preserved_priorities: List[ContextPriority]
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "original_items": self.original_items,
            "summarized_items": self.summarized_items,
            "compression_ratio": self.compression_ratio,
            "key_points": self.key_points,
            "preserved_priorities": [p.value for p in self.preserved_priorities],
            "created_at": self.created_at.isoformat()
        }


class ContextMergeStrategy(Enum):
    """Strategies for merging contexts"""
    PRIORITY_BASED = "priority"      # Merge based on priority levels
    TIMESTAMP_BASED = "timestamp"    # Newer values override older
    WEIGHTED = "weighted"           # Use relevance scores as weights
    CONSERVATIVE = "conservative"   # Keep all, resolve conflicts conservatively
    AGGRESSIVE = "aggressive"       # Prefer local context, filter heavily