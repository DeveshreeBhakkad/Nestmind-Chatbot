from chatnode import ChatNode

# Create root node
root = ChatNode(state="start", message="Hello! How can I help you today?")

# Create child nodes
info_node = ChatNode(state="info", message="I can provide information about NestMind.")
faq_node = ChatNode(state="faq", message="Here are some frequently asked questions.")
bye_node = ChatNode(state="end", message="Goodbye! Have a great day!")

# Add children
root.add_child(info_node)
root.add_child(faq_node)
root.add_child(bye_node)

# Add child to info_node
details_node = ChatNode(state="details", message="NestMind is an AI-powered learning assistant.")
info_node.add_child(details_node)

# Traverse and display tree
root.traverse()
