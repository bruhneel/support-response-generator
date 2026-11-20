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


# Milestone 2: Implement the n-gram model with smoothing.
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Create a text generation function to suggest response continuations.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Build a command-line interface for agents to get suggestions.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")

