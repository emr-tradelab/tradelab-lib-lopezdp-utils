# CLAUDE.md

This file provides guidance to Claude Code when working with this Tradelab library template.

## Project Overview

This is a **template repository** for creating new Tradelab libraries. It provides standardized project structure, CI/CD workflows, release automation, and development tooling.

**When creating a new library from this template:**
1. Replace all instances of `mylibrary` with your library name
2. Update `pyproject.toml` with correct package name and dependencies
3. Update this CLAUDE.md with library-specific guidance
4. Implement your library code in `src/tradelab/<library-name>/`
5. Add tests in `tests/`

## Module Structure

```
tradelab-lib-<name>/
├── src/tradelab/<library-name>/
│   ├── __init__.py              # Public API exports
│   └── ...                      # Your modules here
├── tests/
│   ├── conftest.py              # Pytest configuration and fixtures
│   └── test_*.py                # Test modules
├── .github/workflows/
│   ├── quality.yml              # Code quality checks on every push
│   └── release.yml              # Automated publishing to GAR
├── pyproject.toml               # Dependencies and build config
├── config-gar.env               # GCP/GAR configuration
├── release.sh                   # Interactive release script
└── CLAUDE.md                    # This file - update with specific guidance
```

## Development Commands

### Environment Setup
```bash
# Install dependencies
uv sync --all-extras --dev

# Activate virtual environment
source .venv/bin/activate

# Install pre-commit hooks
uv run pre-commit install
```

### Code Quality
```bash
# Auto-fix linting issues and format code
uv run ruff check . --fix
uv run ruff format .

# Check only (no changes)
uv run ruff check .
```

### Testing
```bash
# Run all tests
uv run pytest -v

# Run specific test file
uv run pytest tests/test_example.py

# Run tests with timing
uv run pytest -v --durations=0
```

## Release Process

This template uses a **dual-environment release workflow** (test/prod):

### Test Release (Development)
Publishes to test GAR (`tradelab023`) for validation:

```bash
./release.sh test patch "fix: correct validation"
# Creates tag: v0.1.1-test
# Publishes to: tradelab023 GAR
```

### Production Release
Publishes to production GAR (`tradelab023-pro`) for downstream use:

```bash
./release.sh prod minor "feat: add new feature"
# Creates tag: v0.2.0
# Publishes to: tradelab023-pro GAR
```

**What the script does:**
1. Validates you're on main branch
2. Runs quality checks (`ruff check`)
3. Runs full test suite (`pytest`)
4. Bumps version in `pyproject.toml`
5. Commits changes and creates annotated git tag
6. Pushes to origin (triggers GitHub Actions workflow)

**Tag patterns:**
- `v1.0.0-test` → Test environment (tradelab023)
- `v1.0.0` → Production environment (tradelab023-pro)

### CI/CD Publishing

After pushing a tag, the GitHub Actions workflow (`.github/workflows/release.yml`):
1. Runs quality checks and tests
2. Auto-detects environment from tag suffix
3. Authenticates to GCP via Workload Identity Federation
4. Builds package with `uv build`
5. Publishes to appropriate Google Artifact Registry

**Monitor releases:** `https://github.com/YOUR-ORG/YOUR-REPO/actions`

## Architecture Notes

### Import Guidelines

Libraries should use the `tradelab.<library-name>` namespace:

```python
# Correct - absolute imports
from tradelab.mylibrary.module import MyClass
from tradelab.mylibrary import public_function

# For public API (exported in __init__.py)
from tradelab.mylibrary import MyClass
```

### Dependency Management

- **Runtime dependencies:** Add to `project.dependencies` in `pyproject.toml`
- **Optional dependencies:** Use `project.optional-dependencies` for feature groups
- **Dev dependencies:** Add to `dependency-groups.dev`

Example optional dependency groups:
```toml
[project.optional-dependencies]
feature1 = ["package1>=1.0.0"]
feature2 = ["package2>=2.0.0"]
all = ["package1>=1.0.0", "package2>=2.0.0"]
```

Install with: `uv sync --extra feature1 --dev`

### Testing Patterns

Use pytest with fixtures in `conftest.py`:

```python
# conftest.py
import pytest

@pytest.fixture(scope="session")
def example_fixture():
    return {"key": "value"}

# test_example.py
def test_example(example_fixture):
    assert example_fixture["key"] == "value"
```

## Local Development with Google Artifact Registry

### Installing This Library from GAR

```bash
# Authenticate
export ARTIFACT_REGISTRY_TOKEN=$(gcloud auth application-default print-access-token)
export UV_INDEX_TRADELAB_PYPI_USERNAME=oauth2accesstoken
export UV_INDEX_TRADELAB_PYPI_PASSWORD="$ARTIFACT_REGISTRY_TOKEN"

# Install from GAR (production)
uv pip install tradelab-<library-name> \
  --index-url https://europe-southwest1-python.pkg.dev/tradelab023-pro/tradelab-pypi/simple/

# Install from GAR (test)
uv pip install tradelab-<library-name> \
  --index-url https://europe-southwest1-python.pkg.dev/tradelab023/tradelab-pypi/simple/
```

### Editable Installs for Development

In downstream repositories:

```bash
# From another Tradelab repo (e.g., tradelab-backtester)
uv pip install -e ../tradelab-lib-<name>
```

## Code Standards

Follow Tradelab conventions:

- **Python:** ≥3.12 with type hints
- **Docstrings:** Google style for complex functions
- **Data operations:** Prefer Polars over pandas
- **Validation:** Use Pydantic for data validation
- **Logging:** Use logging module, not print statements
- **Datetime:** Explicit UTC timezone handling
- **Line length:** 100 characters (configured in pyproject.toml)
- **Linting:** Ruff (rules: F, E, W, I, RUF)

## Pre-commit Hooks

The template includes pre-commit hooks that run automatically on `git commit`:

```bash
# Install hooks
uv run pre-commit install

# Run manually on all files
uv run pre-commit run --all-files
```

**Configured hooks:**
- Ruff formatting
- Ruff linting with auto-fix
- Trailing whitespace removal
- EOF fixing
- YAML/TOML validation
- UV lock sync check

## Notes for Claude Code

When working with this template:

1. **Always update this CLAUDE.md** with library-specific guidance
2. **Replace placeholder names** (`mylibrary`, `YOUR-ORG`, etc.)
3. **Add module-specific documentation** in relevant sections
4. **Document architectural decisions** as the library evolves
5. **Keep dependency list current** with actual usage patterns
6. **Update import examples** with actual module names
