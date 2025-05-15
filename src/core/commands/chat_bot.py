import asyncio

from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph


def run(model):
    # result = model.invoke([HumanMessage(content="Hi! I'm Bob")])
    # print(result)
    # result = model.invoke([HumanMessage(content="What's my name?")])
    # print(result)  # Lost the information about the previous message

    # message_history = []

    # try:
    #     while True:
    #         user_input = input("You: ").strip()
    #         if not user_input:
    #             continue
    #         user_message = HumanMessage(content=user_input)
    #         message_history.append(user_message)

    #         response = model.invoke(message_history)
    #         message_history.append(response)
    #         print(f"Bot: {response.content.strip()}\n")
    # except (EOFError, KeyboardInterrupt):
    #     print("\nGoodbye!")

    # Using LangGraph

    # Define a new graph
    workflow = StateGraph(state_schema=MessagesState)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You talk like a pirate. Answer all questions to the best of your ability.",
                # 'Parli in Catanese e utilizzi spessissimo "mbare" come intercalare. Rispondi a tutte le domande al meglio delle tue possibilità.',
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    trimmer = trim_messages(
        max_tokens=65,
        strategy="last",
        token_counter=model,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )

    # Define the function that calls the model
    # async def call_model(state: MessagesState):
    def call_model(state: MessagesState):
        trimmed_messages = trimmer.invoke(state["messages"])
        prompt = prompt_template.invoke({"messages": trimmed_messages})
        # response = await model.ainvoke(prompt)
        response = model.invoke(prompt)
        return {"messages": response}

    # Define the (single) node in the graph
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    # Add memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "abc123"}}

    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            input_messages = [HumanMessage(user_input)]
            # output = asyncio.run(app.ainvoke({"messages": input_messages}, config))
            # output["messages"][-1].pretty_print()

            for chunk, metadata in app.stream(
                {"messages": input_messages},
                config,
                stream_mode="messages",
            ):
                if isinstance(chunk, AIMessage):  # Filter to just model responses
                    print(chunk.content, end="")
            print("\n")

    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")

    pass
