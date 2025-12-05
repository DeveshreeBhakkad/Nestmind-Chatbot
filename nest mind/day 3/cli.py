# cli.py
from conversation_tree import ConversationTree
from chatbot_engine import ChatBotEngine

def run_cli():
    tree = ConversationTree("Welcome to NestMind!")
    bot = ChatBotEngine()

    print("=== NestMind CLI (Day 3) ===")
    print("Commands: ask | new | list | dfs | bfs | exit")
    
    while True:
        cmd = input("\n>> ").strip().lower()
        
        if cmd == "ask":
            question = input("Your question: ")
            answer = bot.get_answer(question)
            
            # Store Q&A in current active node
            tree.active_node.message = question
            tree.active_node.set_answer(answer)
            
            print(f"🤖 Answer: {answer}")
        
        elif cmd == "new":
            msg = input("Enter new sub-question: ")
            node = tree.create_sub_chat(message=msg)
            ans = bot.get_answer(msg)
            node.set_answer(ans)
            print(f"✅ Sub-chat created: Q: {msg} → A: {ans}")
        
        elif cmd == "list":
            print("\n🌳 Conversation Tree:")
            tree.list_chats()
        
        elif cmd == "dfs":
            print("\nDFS Traversal:")
            for node in tree.dfs():
                print(f"- Q: {node.message} | A: {node.answer}")
        
        elif cmd == "bfs":
            print("\nBFS Traversal:")
            for node in tree.bfs():
                print(f"- Q: {node.message} | A: {node.answer}")
        
        elif cmd == "exit":
            print("👋 Exiting NestMind CLI...")
            break
        
        else:
            print("⚠️ Unknown command. Try again.")

if __name__ == "__main__":
    run_cli()
