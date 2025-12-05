# conversation_tree.py
from chat_node import ChatNode

class ConversationTree:
    def __init__(self, root_message="Main Problem"):
        self.root = ChatNode(message=root_message)
        self.active_node = self.root

    def create_sub_chat(self, message="New Sub Chat", parent=None):
        """Add a sub-chat under the active node or a given parent."""
        parent = parent or self.active_node
        return parent.add_child(message)

    def switch_node(self, node):
        """Switch active conversation to another node."""
        self.active_node = node

    def list_chats(self, node=None, level=0):
        """Print the conversation tree structure with Q&A."""
        node = node or self.root
        prefix = " " * level
        ans = f" → {node.answer}" if node.answer else ""
        print(f"{prefix}- Q: {node.message}{ans} (State: {node.state})")
        for child in node.children:
            self.list_chats(child, level + 2)

    def dfs(self, node=None):
        """Depth-first traversal."""
        node = node or self.root
        yield node
        for child in node.children:
            yield from self.dfs(child)

    def bfs(self, node=None):
        """Breadth-first traversal."""
        node = node or self.root
        queue = [node]
        while queue:
            current = queue.pop(0)
            yield current
            queue.extend(current.children)

