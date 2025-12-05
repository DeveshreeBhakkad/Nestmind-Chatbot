from typing import Optional, Dict

class ContextManager:
    """Manages context across conversation nodes."""
    def __init__(self):
        self.global_context: Dict = {}

    def set_global_context(self, key, value):
        self.global_context[key] = value

    def merge_contexts(self, node, parent_node: Optional[object] = None) -> Dict:
        merged = dict(self.global_context)
        if parent_node:
            merged.update(parent_node.context)
        merged.update(node.context)
        return merged

    def get_context_summary(self, node) -> Dict:
        return {
            "node_id": node.id,
            "local_context": node.context,
            "merged_context": self.merge_contexts(node)
        }
