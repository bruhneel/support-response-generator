"""Preprocessing, language modeling, and sampling for support replies."""
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
import math
import random
import re
import unicodedata

START, END, UNK = "<s>", "</s>", "<UNK>"
PLACEHOLDERS = ("<EMAIL>", "<PHONE>", "<ORDER_ID>", "<URL>", "<NAME>")
NORMALIZE = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "—": " ", "–": " "})
REDACTIONS = (
    (re.compile(r"\b(?:https?://|www\.)[^\s<>]+", re.I), "<URL>"),
    (re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.I), "<EMAIL>"),
    (re.compile(r"\b(?:ORD|ORDER)[-#]\s*[A-Z0-9]*\d[A-Z0-9-]*\b|(?<=order )#\s*[A-Z0-9]*\d[A-Z0-9-]*\b", re.I), "<ORDER_ID>"),
    (re.compile(r"(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}(?!\w)"), "<PHONE>"),
)
NAME = re.compile(r"(\b(?i:my name is|i am|this is) |\b(?i:hi|hello|dear) +)([A-Z][a-z]+ [A-Z][a-z]+)(?=[,!.?]|$)")
TOKENS = re.compile(r"<(?:EMAIL|PHONE|ORDER_ID|URL|NAME)>|[^\W_]+(?:'[^\W_]+)*|[.!?]+", re.I)


def anonymize(text: str) -> str:
    """Mask common identifiers and names in explicit introductions or greetings."""
    text = unicodedata.normalize("NFKC", text).translate(NORMALIZE)
    for pattern, placeholder in REDACTIONS:
        if placeholder == "<URL>":
            text = pattern.sub(
                lambda match: placeholder + match[0][len(match[0].rstrip(".,!?;):")):], text
            )
        else:
            text = pattern.sub(placeholder, text)
    return NAME.sub(r"\1<NAME>", text)


def tokenize(text: str) -> list[str]:
    """Lowercase words, preserve placeholders and retain sentence terminators."""
    return [
        token.upper() if token.startswith("<") else token.lower()
        for token in TOKENS.findall(anonymize(text))
    ]


def sentences(text: str) -> list[list[str]]:
    """Return nonempty sentences with one start and end marker each."""
    output = []
    words = []
    for token in tokenize(text):
        if token[0] in ".!?":
            if words:
                output.append([START, *words, END])
                words = []
        else:
            words.append(token)
    if words:
        output.append([START, *words, END])
    return output


class NGramModel:
    """Count suffix contexts and apply add-k smoothing to the longest seen one."""

    def __init__(self, n: int = 3, k: float = 1.0) -> None:
        if not 1 <= n <= 20:
            raise ValueError("N-gram order must be between 1 and 20.")
        if not math.isfinite(k) or k <= 0:
            raise ValueError("Smoothing k must be positive and finite.")
        self.n = n
        self.k = k
        self.vocab: tuple[str, ...] = ()
        self.counts: dict[tuple[str, ...], Counter[str]] = {}
        self._vocabulary: set[str] = set()
        self._totals: dict[tuple[str, ...], int] = {}

    def fit(self, prepared: Iterable[Sequence[str]]) -> "NGramModel":
        """Count all orders up to n, padding contexts with sentence starts."""
        counts: defaultdict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        vocab = {UNK, END}
        for sentence in prepared:
            if len(sentence) < 3 or sentence[0] != START or sentence[-1] != END:
                raise ValueError("Training sentences must have start/end markers and at least one word.")
            context = (START,) * (self.n - 1)
            for word in sentence[1:]:
                vocab.add(word)
                for size in range(min(self.n - 1, len(context)) + 1):
                    counts[context[len(context) - size:]][word] += 1
                context = (*context, word)[1:] if self.n > 1 else ()
        if not counts:
            raise ValueError("Training needs at least one sentence.")
        self._install(tuple(sorted(vocab)), dict(counts))
        return self

    def _install(self, vocab: tuple[str, ...], counts: dict[tuple[str, ...], Counter[str]]) -> None:
        self.vocab = vocab
        self._vocabulary = set(vocab)
        self.counts = counts
        self._totals = {context: row.total() for context, row in counts.items()}

    def _backoff(self, context: Sequence[str]) -> tuple[str, ...]:
        if not self.vocab:
            raise ValueError("Train the model before requesting probabilities.")
        suffix = tuple(
            word if word == START or word in self._vocabulary else UNK
            for word in context[max(0, len(context) - self.n + 1):]
        )
        while suffix and suffix not in self.counts:
            suffix = suffix[1:]
        return suffix

    def distribution(self, context: Sequence[str]) -> dict[str, float]:
        """Return a normalized next-token distribution over the trained vocabulary."""
        suffix = self._backoff(context)
        row = self.counts[suffix]
        denominator = self._totals[suffix] + self.k * len(self.vocab)
        return {word: (row[word] + self.k) / denominator for word in self.vocab}

    def prob(self, word: str, context: Sequence[str]) -> float:
        """Return an add-k probability, mapping out-of-vocabulary words to UNK."""
        suffix = self._backoff(context)
        if word == START:
            return 0.0
        target = word if word in self._vocabulary else UNK
        return (self.counts[suffix][target] + self.k) / (
            self._totals[suffix] + self.k * len(self.vocab)
        )

    def to_record(self) -> dict:
        """Produce a JSON-compatible representation of counts and hyperparameters."""
        return {
            "n": self.n,
            "k": self.k,
            "vocab": list(self.vocab),
            "counts": [[list(context), dict(row)] for context, row in sorted(self.counts.items())],
        }

    @classmethod
    def from_record(cls, record: dict) -> "NGramModel":
        """Reconstruct a saved n-gram model."""
        model = cls(record["n"], record["k"])
        model._install(
            tuple(record["vocab"]),
            {tuple(context): Counter(row) for context, row in record["counts"]},
        )
        return model


def _prompt_context(partial_text: str, order: int) -> tuple[str, ...]:
    words = [START] * (order - 1)
    for token in tokenize(partial_text):
        if token[0] in ".!?":
            words = [START] * (order - 1)
        else:
            words.append(token)
    return tuple(words[max(0, len(words) - order + 1):])


def _sample(model: NGramModel, context: Sequence[str], rng: random.Random, temperature: float) -> str:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite.")
    probabilities = model.distribution(context)
    peak = math.log(max(probabilities.values()))
    weights = [math.exp((math.log(p) - peak) / temperature) for p in probabilities.values()]
    return rng.choices(model.vocab, weights=weights, k=1)[0]


def suggest_next(
    partial_text: str, model: NGramModel, *, seed: int | None = None, temperature: float = 1.0
) -> str:
    """Sample one token given the final words of the active sentence."""
    return _sample(model, _prompt_context(partial_text, model.n), random.Random(seed), temperature)


def continue_response(
    partial_text: str, model: NGramModel, max_words: int = 16, *,
    seed: int | None = None, temperature: float = 1.0
) -> str:
    """Return only the drafted suffix, stopping at sentence end or the word limit."""
    if max_words < 0:
        raise ValueError("The word limit must be non-negative.")
    rng = random.Random(seed)
    context = _prompt_context(partial_text, model.n)
    draft: list[str] = []
    for _ in range(max_words):
        word = _sample(model, context, rng, temperature)
        if word == END:
            return " ".join(draft) + ("." if draft else "")
        draft.append(word)
        if model.n > 1:
            context = (*context, word)[-(model.n - 1):]
    return " ".join(draft)
