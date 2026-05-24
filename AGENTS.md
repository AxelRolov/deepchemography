# Repository Guidelines

## Project Structure & Module Organization

DeepChemography uses a Python `src/` layout. Package code lives in `src/deepchemography/`: `smiles/` contains the LSTM autoencoder and SMILES WAE, `peptides/` contains the peptide WAE, and `shared/` contains reusable logging, loss, and utility code. Root-level `utils.py` and `script_utils.py` provide compatibility and CLI helpers. Training entry points are in `scripts/`, longer instructions are in `docs/`, notebooks are in `notebooks/`, and checked-in datasets, outputs, and pretrained checkpoints are under `data/`, `output/`, and `models/`. Tests belong in `tests/`.

## Build, Test, and Development Commands

- `uv sync --extra dev`: install runtime and development dependencies.
- `uv run pytest`: run the test suite used by CI.
- `uv run python scripts/train_autoencoder.py --help`: inspect SMILES autoencoder training options.
- `uv run jupyter notebook notebooks/autoencoder_example.ipynb`: launch the example notebook locally.
- `uv build`: build the package with the Hatchling backend.

Requires Python 3.11 or newer; CI currently runs Python 3.11 and 3.12.

## Coding Style & Naming Conventions

Follow the existing Python style: 4-space indentation, clear module-level imports, descriptive snake_case functions, PascalCase classes, and type hints where practical. Add docstrings to public functions and classes. Keep `smiles` and `peptides` independent except for shared utilities in `src/deepchemography/shared/`. Prefer high-level APIs in each module's `api.py` for user-facing workflows, and keep checkpoint loading compatible with the existing `config.pt`, `vocab.pt`, and `model*.pt` pattern.

## Testing Guidelines

Use `pytest`. Name test files `test_*.py` and place them in `tests/`, mirroring the package area under test when useful. Add focused tests for new behavior, especially tensor shapes, vocabulary conversion, model loading, API round trips, and error handling. Run `uv run pytest` before opening a PR.

## Commit & Pull Request Guidelines

Recent commits use short, plain-English subjects such as `added activity data` and `Modernize repository: ...`; keep subjects concise and descriptive. For pull requests, use the repository template: include a summary, select the change type, confirm tests pass, and update documentation when behavior, training commands, or public APIs change. Link related issues when applicable.

## Data, Models, and Generated Outputs

Do not add large datasets, checkpoints, notebook exports, or `output/` artifacts unless they are intentional project assets. Keep reproducible training instructions in `docs/` or `scripts/` rather than relying on unpublished local state.
