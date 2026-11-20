"""Regression tests for the support response pipeline."""
import pytest

from app.core import END, START, anonymize, sentences, tokenize


@pytest.mark.parametrize(("source", "expected"), [
    ("Email a@example.invalid.", "Email <EMAIL>."),
    ("Call +1 (202) 555-0108.", "Call <PHONE>."),
    ("Open order ORD-123456.", "Open order <ORDER_ID>."),
    ("Order #99123 is ready.", "Order <ORDER_ID> is ready."),
    ("Go to https://example.invalid/help.", "Go to <URL>."),
    ("My name is Avery Example.", "My name is <NAME>."),
])
def test_anonymize(source, expected):
    assert anonymize(source) == expected
    assert anonymize(expected) == expected


def test_tokenize_and_boundaries():
    assert tokenize("  WE’LL   email <email>,  Thanks! ") == [
        "we'll", "email", "<EMAIL>", "thanks", "!"
    ]
    assert sentences("First step?! ... Second step") == [
        [START, "first", "step", END], [START, "second", "step", END]
    ]
    assert sentences("?!") == []


def test_committed_corpus_is_redacted():
    from pathlib import Path

    from data.build_corpus import build_corpus

    corpus = Path("data/support_corpus.txt").read_text()
    assert len(corpus.splitlines()) == 320
    assert corpus == build_corpus() == anonymize(corpus)
    for placeholder in ("<EMAIL>", "<PHONE>", "<ORDER_ID>", "<URL>", "<NAME>"):
        assert corpus.count(placeholder) == 160


def test_counts_smoothing_and_backoff():
    from app.core import NGramModel, UNK

    model = NGramModel().fit(sentences("I can help. I can check."))
    assert model.counts[(START, START)] == {"i": 2}
    assert model.counts[("i", "can")] == {"help": 1, "check": 1}
    assert model.counts[("can", "help")] == {END: 1}
    context = ("i", "can")
    assert sum(model.prob(word, context) for word in model.vocab) == pytest.approx(1)
    assert model.prob("i", context) > 0
    assert model.prob("newword", context) == model.prob(UNK, context)
    assert model.distribution(("absent", "can")) == model.distribution(("can",))
    assert model.prob(START, context) == 0
    restored = NGramModel.from_record(model.to_record())
    assert restored.distribution(context) == model.distribution(context)
    unigram = NGramModel(n=1, k=0.5).fit(sentences("One. Two."))
    assert unigram.prob("one", ("arbitrary",)) == pytest.approx(1.5 / 6)


def test_seeded_suggestions_and_stopping():
    from app.core import NGramModel, continue_response, suggest_next

    model = NGramModel().fit(sentences("We can help. We can check.") * 15)
    prompt = "Thanks!  WE can"
    choice = suggest_next(prompt, model, seed=17)
    assert choice in model.vocab
    assert choice == suggest_next("we can", model, seed=17)
    assert continue_response(prompt, model, seed=17) == continue_response(prompt, model, seed=17)
    assert suggest_next("we can help", model, seed=17, temperature=0.01) == END
    assert continue_response("we can help", model, seed=17, temperature=0.01) == ""
    assert continue_response("we can", model, max_words=1, seed=17, temperature=0.01) in {"help", "check"}
    assert continue_response("we can", model, seed=17, temperature=0.01) in {"help.", "check."}
    assert continue_response("we can", model, max_words=0) == ""
    with pytest.raises(ValueError, match="Temperature"):
        suggest_next(prompt, model, temperature=0)


def test_cli_end_to_end(tmp_path, monkeypatch):
    from pathlib import Path
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("PYTHONPATH", str(root))

    def run(*args, input_text=None):
        return subprocess.run(
            [sys.executable, "-m", "app.cli", *args],
            cwd=tmp_path, input=input_text, text=True, capture_output=True, check=False,
        )

    assert "Model not trained yet" in run("suggest", "we are").stderr
    assert "Run prepare first" in run("train").stderr
    assert "380 sentences" in run("prepare", "--corpus", str(root / "data/support_corpus.txt")).stdout
    assert "137 vocabulary tokens" in run("train").stdout
    prompt = "Thank you for contacting us, we are"
    result = run("suggest", prompt, "--seed", "42")
    assert result.returncode == 0
    assert result.stdout == "Next word: happy\nDraft continuation: happy to check the return options.\n"
    interactive = run("interactive", input_text=f"{prompt}\n/quit\n")
    assert interactive.returncode == 0 and result.stdout in interactive.stdout
    invalid = run("suggest", prompt, "--temperature", "nan")
    assert invalid.returncode == 2 and "positive and finite" in invalid.stderr
