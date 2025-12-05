"""
Context persistence layer for saving/loading context data
"""
# src/nest_mind/context_persistence.py
# src/nest_mind/context_persistence.py

# nest_mind/context_persistence.py

import sqlite3
from typing import Dict, Any
from datetime import datetime
from .core.context_types import ContextItem, ContextPriority, ContextScope
from .utils.logger import logger
from pathlib import Path


class ContextManager:
    """Manages context items, with persistence to SQLite database."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = Path(__file__).parent / "data" / "context.db"
        self.db_path = db_path
        self.global_context: Dict[str, Any] = {}

        # Initialize DB
        self._init_db()
        logger.info(f"Initialized ContextPersistence with DB: {self.db_path}")

    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        # Create table for context items if not exists
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS context (
            node_id TEXT,
            key TEXT,
            value TEXT,
            priority INTEGER,
            scope TEXT,
            created_at TEXT,
            updated_at TEXT,
            PRIMARY KEY (node_id, key)
        )
        """)
        self.conn.commit()

    def add_context(self, node_id: str, value: Any, key: str = None,
                    priority: ContextPriority = ContextPriority.MEDIUM,
                    scope: ContextScope = ContextScope.LOCAL):
        """Add a context item for a node."""
        if key is None:
            key = str(value)

        now = datetime.utcnow().isoformat()
        self.cursor.execute("""
            INSERT OR REPLACE INTO context (node_id, key, value, priority, scope, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (node_id, key, str(value), priority.value, scope.value, now, now))
        self.conn.commit()
        logger.info(f"Added context for node {node_id}: {key} = {value}")

    def get_context(self, node_id: str) -> Dict[str, Any]:
        """Retrieve all context items for a node."""
        self.cursor.execute("SELECT key, value FROM context WHERE node_id = ?", (node_id,))
        rows = self.cursor.fetchall()
        return {key: value for key, value in rows}

    def set_global_context(self, key: str, value: Any):
        """Set a global key-value context."""
        self.global_context[key] = value
        logger.info(f"Set global context: {key} = {value}")

    def inherit_context(self, child_id: str, parent_id: str):
        """Copy all context items from parent node to child node."""
        parent_contexts = self.get_context(parent_id)
        if not parent_contexts:
            return
        for key, value in parent_contexts.items():
            self.add_context(child_id, value, key=key)
        logger.info(f"Inherited {len(parent_contexts)} context items from {parent_id} -> {child_id}")

    def summarize_context(self, node_id: str):
        """Return a list of context keys (simple summary for demo)."""
        context = self.get_context(node_id)
        return list(context.keys())
