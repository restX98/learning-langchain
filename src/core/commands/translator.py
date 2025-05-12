from langchain_core.prompts import ChatPromptTemplate


def run(model):
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

    system_template = 'Translate the following from English into {language}. If the User input is already in italian, just response "Questo è già italiano". If the User input is not in English say "This is not english"'

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
