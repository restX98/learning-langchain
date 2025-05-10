import os
import debugpy

def init_project():
    if "OPENAI_API_KEY" not in os.environ:
        print("Set OPENAI_API_KEY on .env file")

    if os.environ["LANGSMITH_TRACING"] == "true":
        if "LANGSMITH_API_KEY" not in os.environ:
            print("Set LANGSMITH_API_KEY on .env file")
        if "LANGSMITH_PROJECT" not in os.environ:
            print("Set LANGSMITH_PROJECT on .env file")
    pass

def init_debuger():
    if os.environ["DEBUG"] == "true":
        debugpy.listen(("0.0.0.0", 5678))
        print("Waiting for client to attach...")
        debugpy.wait_for_client()
    pass

if __name__ == "__main__":
    print("Learning Langchain is running")

    init_project()
    init_debuger()

    while True:
        try:
            raw = input("> ").strip()
            if not raw:
                continue
            
            print("Input: ", raw)
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

    pass
