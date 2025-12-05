# chat_node.py

class ChatNode:
    def __init__(self, message, parent=None):
        self.message = message            # User's question
        self.answer = None                # Bot's response
        self.parent = parent              # Parent node
        self.children = []                # Sub-chats
        self.state = "Active"             # Active | Paused | Completed

    def add_child(self, message):
        """Create a new sub-chat (child node)."""
        child = ChatNode(message, parent=self)
        self.children.append(child)
        return child

    def set_answer(self, answer):
        """Assign an answer to this node."""
        self.answer = answer

    def __repr__(self):
        return f"<ChatNode: {self.message} | Answer: {self.answer} | State: {self.state}>"
