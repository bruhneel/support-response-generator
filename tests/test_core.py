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


# Milestone 4: Build a command-line interface for agents to get suggestions.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")

