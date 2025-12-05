from flask import Flask, render_template, jsonify, request
from nest_mind.core.conversation_tree import ConversationTree

app = Flask(__name__)
tree = ConversationTree()

# Create demo nodes
root_id = tree.create_node("Main Problem Discussion")
root_node = tree.get_node(root_id)
root_node.add_message("I have a complex problem to solve.")
root_node.add_message("Let's break it down into smaller sub-problems.")

child1_id = tree.create_node("Sub-problem 1: Analysis", parent_id=root_id)
child2_id = tree.create_node("Sub-problem 2: Solution Design", parent_id=root_id)

tree.get_node(child1_id).add_message("Analyzing data and dependencies.")
tree.get_node(child2_id).add_message("Designing architecture and flow.")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/tree", methods=["GET"])
def get_tree():
    return jsonify(tree.to_dict())

@app.route("/api/add_node", methods=["POST"])
def add_node():
    data = request.json
    title = data.get("title")
    parent_id = data.get("parent_id")
    new_id = tree.create_node(title, parent_id)
    return jsonify({"success": True, "new_id": new_id})

if __name__ == "__main__":
    app.run(debug=True)
