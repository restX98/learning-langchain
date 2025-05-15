import os
import debugpy
import argparse
from enum import Enum

from langchain.chat_models import init_chat_model

import core.commands.translator as translator
import core.commands.semantic_search as semantic_search


def init_debugger():
    debugpy.listen(("0.0.0.0", 5678))
    print("Waiting for client to attach...")
    debugpy.wait_for_client()
    pass


init_debugger()


class Models(Enum):
    TRANSLATOR = "translator"
    SEMANTIC_SEARCH = "semantic_search"
    DEFAULT = TRANSLATOR

    def __str__(self):
        return self.value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script that execute various test using LangChain."
    )
    parser.add_argument(
        "--model",
        type=Models,
        choices=Models,
        required=False,
        default=Models.DEFAULT,
        help=f"You can pick one of this models (Default={Models.DEFAULT})",
    )
    return parser.parse_args()


def init_project():
    if "OPENAI_API_KEY" not in os.environ:
        print("Set OPENAI_API_KEY on .env file")

    if os.environ["LANGSMITH_TRACING"] == "true":
        if "LANGSMITH_API_KEY" not in os.environ:
            print("Set LANGSMITH_API_KEY on .env file")
        if "LANGSMITH_PROJECT" not in os.environ:
            print("Set LANGSMITH_PROJECT on .env file")
    pass


if __name__ == "__main__":
    args = parse_args()

    print("Learning Langchain is running")

    init_project()

    model = init_chat_model("gpt-4o-mini", model_provider="openai")

    if args.model == Models.TRANSLATOR:
        print("Running translator")
        translator.run(model)
    elif args.model == Models.SEMANTIC_SEARCH:
        print("Running semantic search")
        semantic_search.run(model)
    else:
        print("Running default model (translator)")
        translator.run(model)
