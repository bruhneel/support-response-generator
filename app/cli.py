"""Command-line entry point for preparing, training, and drafting replies."""
import argparse
import math
from pathlib import Path

from app.core import END, NGramModel, continue_response, suggest_next
from app.storage import load_model, prepare_corpus, train_model

CORPUS = Path("data/support_corpus.txt")
PREPARED = Path("artifacts/sentences.json")
MODEL = Path("artifacts/model.json")


def _positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def _word_limit(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def _add_draft_options(command: argparse.ArgumentParser) -> None:
    command.add_argument("--words", type=_word_limit, default=16, help="maximum continuation words (default: 16)")
    command.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    command.add_argument("--temperature", type=_positive, default=0.35, help="sampling temperature (default: 0.35)")


def _show_draft(prompt: str, model: NGramModel, args: argparse.Namespace) -> None:
    options = {"seed": args.seed, "temperature": args.temperature}
    token = suggest_next(prompt, model, **options)
    draft = continue_response(prompt, model, max_words=args.words, **options)
    print(f"Next word: {token if token != END else '(end of sentence)'}")
    print(f"Draft continuation: {draft or '(none)'}")


def _interactive(model: NGramModel, args: argparse.Namespace) -> None:
    print("Enter a draft prompt. Type /quit to exit.")
    while True:
        try:
            prompt = input("Support> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if prompt == "/quit":
            return
        if prompt:
            _show_draft(prompt, model, args)


def main(argv: list[str] | None = None) -> None:
    """Parse the requested command and dispatch it to the local pipeline."""
    parser = argparse.ArgumentParser(
        prog="support-response-generator",
        description="Draft retail support continuations with a local trigram model.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("hello", help="check that everything is wired up")
    prepare = commands.add_parser("prepare", help="redact and tokenize agent replies")
    prepare.add_argument("--corpus", type=Path, default=CORPUS, help="role-prefixed UTF-8 dialogue file")
    train = commands.add_parser("train", help="train and save the trigram model")
    train.add_argument("--k", type=_positive, default=1.0, help="add-k smoothing strength (default: 1)")
    suggest = commands.add_parser("suggest", help="suggest a word and draft a continuation")
    suggest.add_argument("prompt", help="partial agent response")
    _add_draft_options(suggest)
    _add_draft_options(commands.add_parser("interactive", help="draft replies in a prompt loop"))
    args = parser.parse_args(argv)

    try:
        match args.command:
            case "hello":
                print("Support Response Generator is ready. Open README.md for your milestones.")
            case "prepare":
                replies, count = prepare_corpus(args.corpus, PREPARED)
                print(f"Prepared {count} sentences from {replies} agent replies.")
            case "train":
                if not PREPARED.is_file():
                    parser.error("Corpus not prepared yet. Run prepare first.")
                size = train_model(PREPARED, MODEL, k=args.k)
                print(f"Trained a trigram model with {size} vocabulary tokens.")
            case "suggest" | "interactive":
                if not MODEL.is_file():
                    parser.error("Model not trained yet. Run prepare and train first.")
                model = load_model(MODEL)
                if args.command == "suggest":
                    _show_draft(args.prompt, model, args)
                else:
                    _interactive(model, args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
