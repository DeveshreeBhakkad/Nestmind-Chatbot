"""
main.py - Nest Mind
Integrated demonstration of simple ChatNode traversal and advanced ConversationTree functionality.
"""
# main.py (inside src)

# File: nest_mind/main.py

import sys
import json
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree

from nest_mind.core.chat_node import ChatNode
from nest_mind.core.conversation_tree import ConversationTree
from nest_mind.context_persistence import ContextManager
from nest_mind.utils.logger import logger
from nest_mind.config import Config

console = Console()
context_manager = ContextManager()


def demo_day5_features():
    console.print(Panel.fit("🧠 [bold green]Nest Mind Demo - Day 5[/bold green]", subtitle="Version 0.1.0"))

    # Set global context
    context_manager.set_global_context("session_id", "demo-session-001")
    context_manager.set_global_context("user_name", "Demo User")
    console.log("Starting Nest Mind Demo - Day 5")

    # Initialize conversation tree
    tree = ConversationTree()

    # -------------------- CREATE NODES --------------------
    root_id = tree.create_node("Main Problem Discussion")
    root_node = tree.get_node(root_id)
    root_node.add_message("Hello, I have a complex problem to solve.")
    root_node.add_message("I'd be happy to help! Can you break down the problem?")
    root_node.add_message("It involves multiple steps and sub-problems.")

    child1_id = tree.create_node("Sub-problem 1: Analysis", parent_id=root_id)
    child2_id = tree.create_node("Sub-problem 2: Solution Design", parent_id=root_id)
    child1 = tree.get_node(child1_id)
    child2 = tree.get_node(child2_id)

    child1.add_message("Analyzing data and dependencies.")
    child1.add_message("Identifying bottlenecks.")

    child2.add_message("Designing solution architecture.")
    child2.add_message("Defining modules and interfaces.")

    subchild1_id = tree.create_node("Sub-analysis: Data Cleaning", parent_id=child1_id)
    subchild1 = tree.get_node(subchild1_id)
    subchild1.add_message("Cleaning missing values and outliers.")

    # -------------------- DISPLAY TREE --------------------
    console.print("\n[bold blue]Conversation Tree:[/bold blue]")
    for root in [tree.get_node(rid) for rid in [root_id]]:
        root.display()

    # -------------------- CONTEXT SUMMARIES --------------------
    console.print("\n[bold magenta]Context Summaries:[/bold magenta]")
    for node in tree.all_nodes():
        node.show_context()

    # -------------------- TREE STATS --------------------
    console.print("\n[bold yellow]Tree Stats:[/bold yellow]")
    stats = tree.get_tree_stats()
    for k, v in stats.items():
        console.print(f"{k}: {v}")

    # -------------------- SEARCH EXAMPLE --------------------
    console.print("\n[bold cyan]Search Example:[/bold cyan]")
    keyword = "data"
    results = tree.search_nodes(keyword)
    console.print(f"Nodes containing keyword '{keyword}': {[node.title for node in results]}")

    # -------------------- EXPORT TREE --------------------
    export_file = "conversation_tree_export.json"
    tree.export_tree(export_file)
    console.print(f"\n[green]Tree exported to:[/green] {export_file}")

    # -------------------- IMPORT TREE --------------------
    console.print("\n[bold cyan]Import Tree Example:[/bold cyan]")
    with open(export_file, "r") as f:
        data = json.load(f)

    imported_root = ChatNode.import_node(data)
    console.print("[green]Imported Tree:[/green]")
    imported_root.display()


if __name__ == "__main__":
    demo_day5_features()
