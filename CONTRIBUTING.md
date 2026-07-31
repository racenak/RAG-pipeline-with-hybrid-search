# Contributing to RAG Pipeline

Thank you for your interest in contributing!

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/vu/RAG-pipeline-with-hybrid-search.git
   cd RAG-pipeline-with-hybrid-search
   ```

2. Create virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   uv sync
   ```

3. Start infrastructure:
   ```bash
   podman compose up -d
   ```

4. Run tests:
   ```bash
   make test
   ```

## Code Style

- **Linter**: Ruff (configured in `pyproject.toml`)
- **Formatter**: Ruff format
- **Type checker**: Mypy
- Run `make check` before committing

## Testing

- Unit tests: `make test-unit`
- Integration tests: `make test-integration`
- Coverage: `make coverage` (target: 80%+)

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Run `make check && make test`
4. Commit with [Conventional Commits](https://www.conventionalcommits.org/) format
5. Open a PR

## Commit Messages

Use Conventional Commits:
- `feat: add new feature`
- `fix: bug fix`
- `docs: documentation update`
- `test: add tests`
- `refactor: code refactoring`
