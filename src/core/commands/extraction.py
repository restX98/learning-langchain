from typing import Optional, List

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import tool_example_to_messages


class Person(BaseModel):
    """Information about a person."""

    # ^ Doc-string for the entity Person.
    # This doc-string is sent to the LLM as the description of the schema Person,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    name: Optional[str] = Field(default=None, description="The name of the person")
    hair_color: Optional[str] = Field(
        default=None, description="The color of the person's hair if known"
    )
    height_in_meters: Optional[str] = Field(
        default=None, description="Height measured in meters"
    )


class Data(BaseModel):
    """Extracted data about people."""

    # Creates a model so that we can extract multiple entities.
    people: List[Person]


def run(model):
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert extraction algorithm. "
                "Only extract relevant information from the text. "
                "If you do not know the value of an attribute asked to extract, "
                "return null for the attribute's value.",
            ),
            # Please see the how-to about improving performance with
            # reference examples.
            # MessagesPlaceholder('examples'),
            ("human", "{text}"),
        ]
    )

    # structured_llm = model.with_structured_output(schema=Person)
    structured_llm = model.with_structured_output(schema=Data)

    # text = "Alan Smith is 6 feet tall and has blond hair."
    text = "My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me."

    prompt = prompt_template.invoke({"text": text})
    # alan = structured_llm.invoke(prompt)
    data = structured_llm.invoke(prompt)

    # print(alan)
    print(data)

    # Reference examples
    examples = [
        (
            "The ocean is vast and blue. It's more than 20,000 feet deep.",
            Data(people=[]),
        ),
        (
            "Fiona traveled far from France to Spain.",
            Data(people=[Person(name="Fiona", height_in_meters=None, hair_color=None)]),
        ),
    ]

    messages = []

    for txt, tool_call in examples:
        if tool_call.people:
            # This final message is optional for some providers
            ai_response = "Detected people."
        else:
            ai_response = "Detected no people."
        messages.extend(
            tool_example_to_messages(txt, [tool_call], ai_response=ai_response)
        )
    for message in messages:
        message.pretty_print()

    message_no_extraction = {
        "role": "user",
        "content": "The solar system is large, but earth has only 1 moon.",
    }

    result = structured_llm.invoke([message_no_extraction])
    print("Wrong extraction: ", result)

    # Contain examples of "negatives"
    result = structured_llm.invoke(messages + [message_no_extraction])
    print("Correct extraction: ", result)

    pass
