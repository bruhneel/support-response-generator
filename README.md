# Support Response Generator

[![Built with askThili](https://msrit.ask.thili.ai/api/v1/public/projects/tpfwk73bfu/badge.svg)](https://msrit.ask.thili.ai/b/tpfwk73bfu)

Draft retail customer support replies with a local, standard-library trigram model. The default corpus models eight common support scenarios, including order status, returns, refunds, shipping delays, damage, payments, discounts, and account access.

## Install and run

From the repository root with Python 3.12 and Git:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m app.cli hello
pytest -q
```

Keep `.env` local. The starter's optional Ollama, Groq, and Gemini helper in `app/llm.py` remains available separately. Preparing, training, and drafting with the n-gram CLI works offline without an API key.

## Milestones

1. **Prepare.** Run `python -m app.cli prepare` to redact and tokenize the agent replies into `artifacts/sentences.json`. To regenerate the repository corpus, run `python -m data.build_corpus`. It creates 160 fictional tickets and 160 paired replies in memory, then saves only the redacted lines. For another UTF-8 source with `Customer:` and `Agent:` lines, pass `prepare --corpus PATH`.
2. **Train.** Run `python -m app.cli train` to build and save the trigram model to `artifacts/model.json`. The default is Laplace smoothing (`--k 1`). Use `--k 0.5` for weaker smoothing.
3. **Suggest a continuation.** Run `python -m app.cli suggest "Thank you for contacting us, we are" --seed 42`. The draft ends at a sentence boundary or the `--words` limit. Use `--temperature` to adjust sampling. Lower values favor the most probable words.
4. **Draft interactively.** Run `python -m app.cli interactive`, type a partial reply, and press Enter. Type `/quit` or send EOF to exit. Run `python -m app.cli --help` for the command list.

## Example session

```text
$ python -m app.cli prepare
Prepared 380 sentences from 160 agent replies.

$ python -m app.cli train
Trained a trigram model with 137 vocabulary tokens.

$ python -m app.cli suggest "Thank you for contacting us, we are" --seed 42
Next word: happy
Draft continuation: happy to check the return options.
```

See the [recorded session](results/example-session.txt).

## How it works

The tokenizer normalizes Unicode and punctuation, lowercases words, preserves redaction placeholders, and inserts sentence boundary markers. Training learns from agent replies only. A trigram counts a next token after two context tokens. If that context was never seen, the model tries the final token by itself, then the corpus-wide unigram counts. Start markers establish sentence context but are excluded from the prediction vocabulary.

Without smoothing, any unseen continuation would have zero probability. Add-k assigns each possible token a positive pseudo-count. For a context with total count *N* and vocabulary size *V*, a token with count *c* has probability *(c + k) / (N + kV)*. This keeps the distribution normalized and gives unseen words a path through the reserved `<UNK>` token. The temperature transform is applied only at sampling time.

## Limitations

The corpus is small and templated. Two context words can produce fluent local phrases, but the model cannot follow a full conversation, check orders, or know policy. Review drafts before sending and replace any placeholders. The regex redactor recognizes common identifier formats and names in explicit introductions or greetings. It cannot guarantee anonymization of real tickets, which need a separate privacy review before sharing.
