# Support Response Generator

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/tpfwk73bfu/badge.svg)](https://ask.thili.ai/b/tpfwk73bfu)

This CLI tool, designed for retail customer support agents, leverages a next-word predictor to streamline response drafting. By analyzing a corpus of historical support interactions, it provides intelligent suggestions, helping agents respond to common queries more efficiently.

## What you are building

**Goal.** You will build a command-line interface tool that helps customer support agents quickly draft responses by generating next-word suggestions based on a corpus of past support interactions.

**Done when.** The project is done when you have prepared an anonymized corpus, implemented an n-gram model with smoothing, created a function to suggest response continuations, and built a command-line interface for agents to get these suggestions.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Prepare an anonymized corpus of customer support interactions.**  
      Hint: For preparing the corpus, focus on text preprocessing and tokenization techniques from the drills.
- [ ] **2. Implement the n-gram model with smoothing.**  
      Hint: Implement the n-gram model by counting word sequences and apply a smoothing technique like Laplace smoothing.
- [ ] **3. Create a text generation function to suggest response continuations.**  
      Hint: Create the generation function by sampling from the probability distribution of next words based on the n-gram model.
- [ ] **4. Build a command-line interface for agents to get suggestions.**  
      Hint: Build the CLI using Python's `argparse` or similar, allowing agents to input partial responses and receive suggestions.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow agents to configure the 'n' value for the n-gram model at runtime.
- Implement and compare another smoothing technique for the n-gram model.
- Extend the text generation to suggest the top N most probable next words, not just one.
