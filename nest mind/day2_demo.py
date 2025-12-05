"""
Day 2 Demo - Enhanced Context Management System
"""
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from datetime import datetime, timedelta

from src.nest_mind import ChatNode, ConversationTree, Config
from src.nest_mind.core.enhanced_context_manager import EnhancedContextManager
from src.nest_mind.core.context_types import ContextPriority, ContextScope
from src.nest_mind.utils.logger import logger


def day2_comprehensive_demo():
    """Comprehensive demo of Day 2 features"""
    console = Console()
    
    console.print(Panel.fit(
        f"🧠 [bold blue]{Config.PROJECT_NAME}[/bold blue] - Day 2 Enhanced Context Demo\n"
        f"Version: 2.0.0 - Intelligent Context Management",
        border_style="blue"
    ))
    
    # Initialize components
    tree = ConversationTree()
    context_manager = EnhancedContextManager(
        max_context_items=20,
        auto_summarize_threshold=15,
        enable_persistence=True
    )
    
    console.print("\n[bold green]🚀 Setting up complex conversation scenario...[/bold green]")
    
    # Create a complex conversation tree
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Creating conversation tree...", total=None)
        
        # Root: Complex software project
        root_id = tree.create_node("Complex Software Project Planning")
        root_node = tree.get_node(root_id)
        
        # Add root context
        context_manager.set_context_item(
            root_id, "project_name", "Nest Mind AI System",
            priority=ContextPriority.CRITICAL,
            scope=ContextScope.INHERITED,
            keywords=["AI", "system", "project"]
        )
        context_manager.set_context_item(
            root_id, "budget", "$50,000",
            priority=ContextPriority.HIGH,
            scope=ContextScope.INHERITED,
            keywords=["budget", "cost", "financial"]
        )
        context_manager.set_context_item(
            root_id, "deadline", "2024-12-31",
            priority=ContextPriority.HIGH,
            scope=ContextScope.INHERITED,
            keywords=["deadline", "timeline", "schedule"]
        )
        
        # Add messages to root
        root_node.add_message("We need to plan a complex AI system development", "user")
        root_node.add_message("I'll help you break this down into manageable components", "assistant")
        
        progress.update(task, description="Creating architecture discussion...")
        
        # Level 1: Architecture Discussion
        arch_id = tree.create_node("Architecture & Design", parent_id=root_id)
        arch_node = tree.get_node(arch_id)
        
        # Add architecture-specific context
        context_manager.set_context_item(
            arch_id, "architecture_type", "Microservices",
            priority=ContextPriority.HIGH,
            scope=ContextScope.LOCAL,
            keywords=["microservices", "architecture", "design"]
        )
        context_manager.set_context_item(
            arch_id, "tech_stack", {"backend": "Python", "frontend": "React", "database": "PostgreSQL"},
            priority=ContextPriority.MEDIUM,
            scope=ContextScope.INHERITED,
            keywords=["python", "react", "postgresql", "tech", "stack"]
        )
        
        arch_node.add_message("Let's discuss the system architecture", "user")
        arch_node.add_message("We'll use a microservices approach with Python backend", "assistant")
        
        progress.update(task, description="Creating development planning...")
        
        # Level 2: Development Planning
        dev_id = tree.create_node("Development Planning", parent_id=arch_id)
        dev_node = tree.get_node(dev_id)
        
        context_manager.set_context_item(
            dev_id, "team_size", 5,
            priority=ContextPriority.MEDIUM,
            scope=ContextScope.LOCAL,
            keywords=["team", "developers", "resources"]
        )
        context_manager.set_context_item(
            dev_id, "methodology", "Agile/Scrum",
            priority=ContextPriority.LOW,
            scope=ContextScope.LOCAL,
            keywords=["agile", "scrum", "methodology"]
        )
        
        dev_node.add_message("How should we organize the development process?", "user")
        dev_node.add_message("Let's use Agile methodology with 2-week sprints", "assistant")
        
        progress.update(task, description="Creating implementation details...")
        
        # Level 3: Implementation Details
        impl_id = tree.create_node("Core AI Implementation", parent_id=dev_id)
        impl_node = tree.get_node(impl_id)
        
        context_manager.set_context_item(
            impl_id, "ai_model", "GPT-based with custom training",
            priority=ContextPriority.CRITICAL,
            scope=ContextScope.LOCAL,
            keywords=["AI", "GPT", "model", "training"]
        )
        context_manager.set_context_item(
            impl_id, "data_requirements", "100GB training data",
            priority=ContextPriority.HIGH,
            scope=ContextScope.LOCAL,
            keywords=["data", "training", "dataset"]
        )
        
        # Add some temporary context that will expire
        context_manager.set_context_item(
            impl_id, "temp_note", "Remember to check GPU availability",
            priority=ContextPriority.LOW,
            scope=ContextScope.LOCAL,
            keywords=["GPU", "hardware"],
            expires_in=timedelta(seconds=2)  # Will expire quickly for demo
        )
        
        impl_node.add_message("What AI model should we use?", "user")
        impl_node.add_message("A custom-trained GPT model would work best", "assistant")
        
        progress.update(task, description="Adding global context...")
        
        # Set some global context
        context_manager.set_context_item(
            "global", "company_name", "Nest Mind Inc",
            priority=ContextPriority.CRITICAL,
            scope=ContextScope.GLOBAL,
            keywords=["company", "organization"]
        )
        context_manager.set_context_item(
            "global", "development_phase", "Planning",
            priority=ContextPriority.MEDIUM,
            scope=ContextScope.GLOBAL,
            keywords=["phase", "planning", "development"]
        )

    console.print("✅ Complex conversation tree created!\n")
    
    # Demo 1: Context Inheritance
    demo_context_inheritance(console, tree, context_manager)
    
    # Demo 2: Relevance Filtering
    demo_relevance_filtering(console, tree, context_manager)
    
    # Demo 3: Context Summarization
    demo_context_summarization(console, tree, context_manager)
    
    # Demo 4: Persistence
    demo_persistence(console, context_manager)
    
    # Demo 5: Statistics and Analytics
    demo_statistics(console, context_manager)


def demo_context_inheritance(console: Console, tree: ConversationTree, context_manager: EnhancedContextManager):
    """Demo context inheritance features"""
    console.print("[bold green]📊 Demo 1: Context Inheritance[/bold green]")
    
    # Get nodes at different levels
    root_nodes = [tree.get_node(id) for id in tree.root_nodes]
    root_node = root_nodes[0] if root_nodes else None
    
    if not root_node:
        console.print("❌ No root node found")
        return
    
    # Find a deep child node
    deep_node = None
    for node in tree.get_all_nodes():
        if node.depth_level >= 2:
            deep_node = node
            break
    
    if deep_node:
        console.print(f"Analyzing context inheritance for: [cyan]{deep_node.title}[/cyan]")
        
        # Get parent node
        parent_node = tree.get_node(deep_node.parent_id) if deep_node.parent_id else None
        
        # Show context at different levels
        table = Table(title="Context Inheritance Analysis")
        table.add_column("Level", style="cyan")
        table.add_column("Context Items", style="green")
        table.add_column("Sample Keys", style="yellow")
        
        # Global context
        global_count = len(context_manager.global_context)
        global_keys = list(context_manager.global_context.keys())[:3]
        table.add_row("Global", str(global_count), ", ".join(global_keys))
        
        # Parent context
        if parent_node:
            parent_context = context_manager._get_local_context(parent_node.id)
            parent_keys = list(parent_context.keys())[:3]
            table.add_row(f"Parent ({parent_node.title[:20]}...)", 
                         str(len(parent_context)), ", ".join(parent_keys))
        
        # Local context
        local_context = context_manager._get_local_context(deep_node.id)
        local_keys = list(local_context.keys())[:3]
        table.add_row(f"Local ({deep_node.title[:20]}...)", 
                     str(len(local_context)), ", ".join(local_keys))
        
        # Merged context
        merged_context = context_manager.get_merged_context(deep_node, parent_node)
        merged_keys = list(merged_context.keys())[:3]
        table.add_row("Merged Result", str(len(merged_context)), ", ".join(merged_keys))
        
        console.print(table)
        console.print()


def demo_relevance_filtering(console: Console, tree: ConversationTree, context_manager: EnhancedContextManager):
    """Demo context relevance filtering"""
    console.print("[bold green]🎯 Demo 2: Relevance Filtering[/bold green]")
    
    # Get a node with messages
    target_node = None
    for node in tree.get_all_nodes():
        if len(node.messages) > 0:
            target_node = node
            break
    
    if not target_node:
        console.print("❌ No node with messages found")
        return
    
    console.print(f"Testing relevance filtering for: [cyan]{target_node.title}[/cyan]")
    
    # Get conversation messages
    conversation_messages = [msg.content for msg in target_node.messages]
    console.print(f"Conversation messages: {len(conversation_messages)}")
    
    # Get all context before filtering
    parent_node = tree.get_node(target_node.parent_id) if target_node.parent_id else None
    all_context = context_manager.get_merged_context(
        target_node, parent_node, conversation_messages=None
    )
    
    # Get filtered context
    filtered_context = context_manager.get_merged_context(
        target_node, parent_node, conversation_messages=conversation_messages
    )
    
    # Show results
    table = Table(title="Relevance Filtering Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Before Filtering", style="red")
    table.add_column("After Filtering", style="green")
    
    table.add_row("Total Items", str(len(all_context)), str(len(filtered_context)))
    
    if len(all_context) > 0:
        avg_relevance_before = sum(item.relevance_score for item in all_context.values()) / len(all_context)
    else:
        avg_relevance_before = 0
    
    if len(filtered_context) > 0:
        avg_relevance_after = sum(item.relevance_score for item in filtered_context.values()) / len(filtered_context)
    else:
        avg_relevance_after = 0
    
    table.add_row("Avg Relevance", f"{avg_relevance_before:.3f}", f"{avg_relevance_after:.3f}")
    
    # Show top relevant items
    if filtered_context:
        sorted_items = sorted(filtered_context.values(), key=lambda x: x.relevance_score, reverse=True)
        top_items = [f"{item.key} ({item.relevance_score:.3f})" for item in sorted_items[:3]]
        table.add_row("Top Items", "-", ", ".join(top_items))
    
    console.print(table)
    console.print()


def demo_context_summarization(console: Console, tree: ConversationTree, context_manager: EnhancedContextManager):
    """Demo context summarization"""
    console.print("[bold green]📝 Demo 3: Context Summarization[/bold green]")
    
    # Find node with most context
    max_context_node = None
    max_context_count = 0
    
    for node in tree.get_all_nodes():
        local_context = context_manager._get_local_context(node.id)
        if len(local_context) > max_context_count:
            max_context_count = len(local_context)
            max_context_node = node
    
    if not max_context_node or max_context_count == 0:
        console.print("❌ No nodes with sufficient context found")
        return
    
    console.print(f"Testing summarization for: [cyan]{max_context_node.title}[/cyan]")
    
    # Get all context items
    local_context = context_manager._get_local_context(max_context_node.id)
    context_items = list(local_context.values())
    
    # Get conversation messages for relevance
    conversation_messages = [msg.content for msg in max_context_node.messages]
    
    # Test summarization
    summarized_items, summary = context_manager.summarizer.summarize_context(
        context_items, conversation_messages, target_size=5
    )
    
    # Display results
    table = Table(title="Context Summarization Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Original Items", str(summary.original_items))
    table.add_row("Summarized Items", str(summary.summarized_items))
    table.add_row("Compression Ratio", f"{summary.compression_ratio:.2%}")
    table.add_row("Key Points", str(len(summary.key_points)))
    
    console.print(table)
    
    if summary.key_points:
        console.print("\n[bold]Key Points Preserved:[/bold]")
        for i, point in enumerate(summary.key_points, 1):
            console.print(f"  {i}. {point}")
    
    console.print()


def demo_persistence(console: Console, context_manager: EnhancedContextManager):
    """Demo context persistence"""
    console.print("[bold green]💾 Demo 4: Context Persistence[/bold green]")
    
    if not context_manager.persistence:
        console.print("❌ Persistence not enabled")
        return
    
    # Show database stats
    stats = context_manager.persistence.get_database_stats()
    
    table = Table(title="Context Database Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Items", str(stats.get('total_items', 0)))
    table.add_row("Unique Nodes", str(stats.get('unique_nodes', 0)))
    table.add_row("Avg Relevance", f"{stats.get('avg_relevance', 0):.3f}")
    table.add_row("Recent Updates", str(stats.get('recent_updates', 0)))
    
    console.print(table)
    
    # Test cleanup
    console.print("\n[bold]Testing cleanup of expired items...[/bold]")
    import time
    time.sleep(3)  # Wait for temp items to expire
    
    cleaned_count = context_manager.cleanup_expired_context()
    console.print(f"✅ Cleaned up {cleaned_count} expired items")
    console.print()


def demo_statistics(console: Console, context_manager: EnhancedContextManager):
    """Demo context statistics and analytics"""
    console.print("[bold green]📈 Demo 5: Context Statistics & Analytics[/bold green]")
    
    stats = context_manager.get_context_statistics()
    
    # Main statistics table
    main_table = Table(title="Context Management Statistics")
    main_table.add_column("Category", style="cyan")
    main_table.add_column("Value", style="green")
    
    main_table.add_row("Global Context Items", str(stats.get('global_context_items', 0)))
    main_table.add_row("Local Context Items", str(stats.get('local_context_items', 0)))
    main_table.add_row("Total Context Items", str(stats.get('total_context_items', 0)))
    main_table.add_row("Nodes with Context", str(stats.get('nodes_with_context', 0)))
    main_table.add_row("Auto-Summarize Threshold", str(stats.get('auto_summarize_threshold', 0)))
    
    console.print(main_table)
    
    # Priority distribution
    if 'priority_distribution' in stats:
        priority_table = Table(title="Context Priority Distribution")
        priority_table.add_column("Priority", style="cyan")
        priority_table.add_column("Count", style="green")
        
        for priority, count in stats['priority_distribution'].items():
            priority_table.add_row(priority, str(count))
        
        console.print("\n")
        console.print(priority_table)
    
    console.print(f"\n✅ [bold green]Day 2 Enhanced Context Management Demo Complete![/bold green]")
    console.print(f"🚀 Your Nest Mind system now has intelligent context management!")


if __name__ == "__main__":
    import sys
    
    # Check if required packages are available
    try:
        import nltk
        import sklearn
        import textblob
        print("✅ All Day 2 dependencies are available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install: pip install nltk scikit-learn textblob")
        sys.exit(1)
    
    # Download required NLTK data
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
    except:
        pass
    
    day2_comprehensive_demo()