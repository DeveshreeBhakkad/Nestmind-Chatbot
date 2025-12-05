from .chat_node import ChatNode
import json


class ConversationTree:
    def __init__(self):
        self.root = None
        self.nodes = {}
        self.active_node_id = None

    def create_node(self, title, parent_id=None):
        parent_node = self.nodes.get(parent_id) if parent_id else None
        node = ChatNode(title, parent_node)
        self.nodes[node.id] = node
        if parent_node:
            parent_node.children.append(node)
        if not self.root:
            self.root = node
        self.active_node_id = node.id
        return node.id

    def get_node(self, node_id):
        return self.nodes.get(node_id)

    def all_nodes(self):
        return list(self.nodes.values())

    def get_tree_stats(self):
        def depth(node):
            if not node.children:
                return 0
            return 1 + max(depth(child) for child in node.children)
    # nest_mind/core/conversation_tree.py.

    def get_node(self, node_id):
        """Return the node by ID"""
        return self.nodes.get(node_id)


        return {
            "total_nodes": len(self.nodes),
            "root_nodes": 1 if self.root else 0,
            "max_depth": depth(self.root) if self.root else 0,
            "active_node": self.active_node_id
        }
