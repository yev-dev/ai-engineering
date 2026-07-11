import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check prompt connectivity to a ChatOllama model."
    )
    parser.add_argument(
        "--model",
        default="deepseek-r1:1.5b",
        help="Ollama model name to test.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Ollama server base URL.",
    )
    parser.add_argument(
        "--prompt",
        default="Summarize why reliable cash flow matters for a business in one sentence.",
        help="Prompt to send through the prompt template and ChatOllama chain.",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=128,
        help="Maximum number of tokens to generate.",
    )
    return parser.parse_args()


def build_chain(model_name: str, base_url: str, num_predict: int):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a concise assistant. Respond clearly and briefly.",
            ),
            ("human", "{user_prompt}"),
        ]
    )
    model = ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=0,
        reasoning=False,
        num_predict=num_predict,
    )
    return prompt | model | StrOutputParser()


def main() -> int:
    args = parse_args()
    chain = build_chain(args.model, args.base_url, args.num_predict)

    try:
        response = chain.invoke({"user_prompt": args.prompt})
    except Exception as exc:
        error_text = str(exc)
        print("ChatOllama prompt connection check failed.", file=sys.stderr)
        print(f"Model: {args.model}", file=sys.stderr)
        print(f"Base URL: {args.base_url}", file=sys.stderr)
        print(f"Error: {error_text}", file=sys.stderr)
        if "not found" in error_text.lower():
            print(
                f"Hint: pull the model first with: ollama pull {args.model}",
                file=sys.stderr,
            )
        elif "connection" in error_text.lower() or "refused" in error_text.lower():
            print(
                "Hint: make sure the Ollama server is running and the base URL is correct.",
                file=sys.stderr,
            )
        return 1

    print("ChatOllama prompt connection check succeeded.")
    print(f"Model: {args.model}")
    print(f"Base URL: {args.base_url}")
    print("Prompt:")
    print(args.prompt)
    print("Response:")
    print(response.strip() or "<empty response>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())