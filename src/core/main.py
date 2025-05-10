import os
import sys
import debugpy

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

def init_project():
    if "OPENAI_API_KEY" not in os.environ:
        print("Set OPENAI_API_KEY on .env file")

    if os.environ["LANGSMITH_TRACING"] == "true":
        if "LANGSMITH_API_KEY" not in os.environ:
            print("Set LANGSMITH_API_KEY on .env file")
        if "LANGSMITH_PROJECT" not in os.environ:
            print("Set LANGSMITH_PROJECT on .env file")
    pass

def init_debugger():
    if os.environ["DEBUG"] == "true":
        debugpy.listen(("0.0.0.0", 5678))
        print("Waiting for client to attach...")
        debugpy.wait_for_client()
    pass

if __name__ == "__main__":
    print("Learning Langchain is running")

    init_project()
    init_debugger()

    model = init_chat_model("gpt-4o-mini", model_provider="openai")

    # messages = [
    #     SystemMessage("Translate the following from English into Italian"),
    #     HumanMessage("hi!"),
    # ]

    # result = model.invoke(messages)

    # print(model)
    # print("\n")
    # print(result)
    # print("\n")

    # for token in model.stream(messages):
    #     print(token.content, end="|")

    # print("\n")

    system_template = "Translate the following from English into {language}. If the User input is already in italian, just response \"Questo è già italiano\". If the User input is not in English say \"This is not english\""

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )

    language = "Italian"

    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            prompt = prompt_template.invoke({"language": language, "text": user_input})
            response = model.invoke(prompt)
            content = response.content.strip()
            print(f"Bot: {content}\n")
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        sys.exit(0)
