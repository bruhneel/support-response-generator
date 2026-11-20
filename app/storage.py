"""A tiny key-value store in one JSON file (no server, easy to read). Values can be anything JSON can hold."""
import json
import os
from pathlib import Path


class Store:
    def __init__(self, path: str | None = None):
        self.path = Path(path or os.environ.get("DATA_PATH", "data/app.json"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}")

    def _read(self) -> dict:
        return json.loads(self.path.read_text() or "{}")

    def put(self, key: str, value) -> None:
        data = self._read()
        data[key] = value
        self.path.write_text(json.dumps(data, indent=2))

    def get(self, key: str, default=None):
        return self._read().get(key, default)

    def all(self) -> dict:
        return dict(sorted(self._read().items()))


def prepare_corpus(source: Path, destination: Path) -> tuple[int, int]:
    """Tokenize anonymized agent replies from role-prefixed support tickets."""
    from app.core import sentences

    lines = source.read_text(encoding="utf-8").splitlines()
    replies = [
        line.split(":", 1)[1]
        for line in lines
        if line.partition(":")[0].strip().lower() == "agent"
    ]
    prepared = [tokens for reply in replies for tokens in sentences(reply)]
    if not prepared:
        raise ValueError("Corpus needs at least one nonempty Agent: line.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(prepared, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(replies), len(prepared)


def train_model(source: Path, destination: Path, n: int = 3, k: float = 1.0) -> int:
    """Fit a model from prepared tokens and save its counts as JSON."""
    from app.core import NGramModel

    prepared = json.loads(source.read_text(encoding="utf-8"))
    model = NGramModel(n=n, k=k).fit(prepared)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(model.to_record(), ensure_ascii=False) + "\n", encoding="utf-8")
    return len(model.vocab)


def load_model(path: Path) -> "NGramModel":
    """Read a saved n-gram model."""
    from app.core import NGramModel

    return NGramModel.from_record(json.loads(path.read_text(encoding="utf-8")))
