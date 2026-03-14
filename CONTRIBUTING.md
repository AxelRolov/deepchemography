# Contributing to DeepChemography

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/AxelRolov/deepchemography.git
cd deepchemography
uv sync --extra dev
```

## Running Tests

```bash
uv run pytest
```

## Code Style

- Follow existing patterns in the codebase
- Keep modules independent (smiles and peptides don't share code except via `shared/`)
- Use type hints where practical
- Add docstrings to public functions

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Ensure tests pass: `uv run pytest`
4. Open a pull request using the PR template
5. Wait for review

## Reporting Issues

Use the [issue templates](https://github.com/AxelRolov/deepchemography/issues/new/choose) to report bugs or request features.
