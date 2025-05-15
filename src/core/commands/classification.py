from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class Classification(BaseModel):
    sentiment: str = Field(
        description="The sentiment of the text", enum=["happy", "neutral", "sad"]
    )
    aggressiveness: int = Field(
        description="Describes how aggressive the statement is, the higher the number the more aggressive",
        enum=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    )
    language: str = Field(
        description="The language the text is written in",
        enum=["spanish", "english", "french", "german", "italian"],
    )


def run(model):
    tagging_prompt = ChatPromptTemplate.from_template(
        """
    Extract the desired information from the following passage.

    Only extract the properties mentioned in the 'Classification' function.

    Passage:
    {input}
    """
    )

    structured_llm = model.with_structured_output(Classification)

    # Input examples:
    # Sono incredibilmente felice di averti incontrato! Penso che diventeremo ottimi amici.
    # Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!
    # Sono così arrabbiata con te! Ti darò ciò che meriti!
    # Estoy muy enojado con vos! Te voy a dar tu merecido!
    # Weather is ok here, I can go outside without much more than a coat

    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            prompt = tagging_prompt.invoke({"input": user_input})

            response = structured_llm.invoke(prompt)

            print(f"Bot: {response}\n")
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")

    pass
