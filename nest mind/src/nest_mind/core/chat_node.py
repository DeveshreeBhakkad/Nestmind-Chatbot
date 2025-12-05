import uuid

class ChatNode:
    def __init__(self, title, parent=None):
        self.id = str(uuid.uuid4())
        self.title = title
        self.parent = parent
        self.children = []
        self.context = []

    def add_context(self, text):
        self.context.append(text)

    def __repr__(self):
        return f"ChatNode({self.title})"
