# retrieval.py
def get_context_for_query(query):
    """
    Temporary: Pretend we searched Day-2 database.
    You can connect your TF-IDF or SQLite later.
    """
    sample_contexts = {
        "ai": "Artificial Intelligence helps machines think and learn.",
        "python": "Python is a popular language for AI and data science.",
        "nlp": "Natural Language Processing helps understand human text."
    }
    for key in sample_contexts:
        if key in query.lower():
            return sample_contexts[key]
    return "No specific context found. Answer generally."
