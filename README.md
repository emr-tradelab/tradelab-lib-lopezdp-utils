# tradelab-lib-template

A standardized Python library template for the Tradelab ecosystem with automated release workflows, quality checks, and dual-environment publishing (test/prod).

## Quick Start

1. **Clone this template** for your new library
2. **Replace placeholder names**:
   - `mylibrary` → your library name in `pyproject.toml` and `src/tradelab/`
   - Update `CLAUDE.md` with library-specific guidance
3. **Install dependencies**: `uv sync --all-extras --dev`
4. **Install pre-commit hooks**: `uv run pre-commit install`
5. **Develop** your library in `src/tradelab/<library-name>/`
6. **Write tests** in `tests/`

## Release Process

This template uses a **dual-environment release workflow** to safely promote libraries from test to production.

### Test Release (Development/Validation)

Publishes to **test GAR** (`tradelab023`) for validation:

```bash
chmod +x release.sh  # First time only
./release.sh test patch "fix: correct validation logic"
```

**Creates tag:** `v0.1.1-test`
**Publishes to:** `tradelab023` (test GAR)

### Production Release

Publishes to **production GAR** (`tradelab023-pro`) for downstream use:

```bash
./release.sh prod minor "feat: add new feature"
```

**Creates tag:** `v0.2.0`
**Publishes to:** `tradelab023-pro` (production GAR)

### What the Script Does

The `release.sh` script automates the entire release process:

1. **Safety checks:**
   - Validates you're on `main` branch
   - Checks for uncommitted changes
   - Verifies you're in sync with remote
2. **Quality gates:**
   - `uv sync --all-extras --dev` - Install dependencies
   - `uv run ruff check .` - Lint checks
   - `uv run pytest -v` - Full test suite
3. **Version management:**
   - Bumps version in `pyproject.toml` (using `uv version --bump`)
   - Creates git commit with changes
   - Creates annotated git tag (`v1.0.0` or `v1.0.0-test`)
4. **Publishing:**
   - Pushes branch and tag to origin
   - Triggers GitHub Actions workflow

**Usage:**
```bash
./release.sh <test|prod> [patch|minor|major] ["commit message"]

# Examples:
./release.sh test patch "fix: resolve validation bug"
./release.sh prod minor "feat: add new API endpoint"
./release.sh prod major "breaking: redesign public interface"
```

## CI/CD (GitHub Actions)

The template includes automated workflows at `.github/workflows/`:

### Quality Workflow (`quality.yml`)

**Triggers:** Every push and pull request
**Runs:**
- Code formatting check (`ruff format`)
- Linting with auto-fix (`ruff check --fix`)
- Full test suite (`pytest -v --durations=0`)

### Release Workflow (`release.yml`)

**Triggers:** Git tags matching patterns:
- `v[0-9]+.[0-9]+.[0-9]+` (e.g., `v1.0.0`) → Production GAR
- `v[0-9]+.[0-9]+.[0-9]+-test` (e.g., `v1.0.0-test`) → Test GAR

**Process:**
1. Runs quality checks and tests
2. Auto-detects environment from tag suffix (`-test` or no suffix)
3. Loads appropriate GCP project from `config-gar.env`
4. Authenticates to GCP via Workload Identity Federation (WIF)
5. Builds package with `uv build`
6. Publishes to Google Artifact Registry

**Monitor releases:** `https://github.com/YOUR-ORG/YOUR-REPO/actions`

## Configuration

### `config-gar.env`

Defines GCP projects for both environments (no manual editing needed):

```bash
# Test environment
TEST_PROJECT_ID="tradelab023"
TEST_PROJECT_NUMBER="566607668180"

# Production environment
PROD_PROJECT_ID="tradelab023-pro"
PROD_PROJECT_NUMBER="607228652441"

# Shared config
REPOSITORY="tradelab-pypi"
GAR_LOCATION="europe-southwest1"
CI_SA="tradelab-pypi-publisher"
```

The release script and GitHub Actions automatically select the correct environment based on your chosen release type (test/prod) or tag suffix.

## Development Workflow

### Local Development

```bash
# Install dependencies
uv sync --all-extras --dev

# Activate virtual environment
source .venv/bin/activate

# Install pre-commit hooks
uv run pre-commit install

# Run formatting
uv run ruff format .

# Run linting
uv run ruff check . --fix

# Run tests
uv run pytest -v
```

### Installing from GAR (Downstream Repos)

```bash
# Authenticate to GAR
export ARTIFACT_REGISTRY_TOKEN=$(gcloud auth application-default print-access-token)
export UV_INDEX_TRADELAB_PYPI_USERNAME=oauth2accesstoken
export UV_INDEX_TRADELAB_PYPI_PASSWORD="$ARTIFACT_REGISTRY_TOKEN"

# Install from production GAR
uv pip install tradelab-<library-name> \
  --index-url https://europe-southwest1-python.pkg.dev/tradelab023-pro/tradelab-pypi/simple/

# Install from test GAR
uv pip install tradelab-<library-name> \
  --index-url https://europe-southwest1-python.pkg.dev/tradelab023/tradelab-pypi/simple/
```

### Editable Installs

For local development across Tradelab repositories:

```bash
# From another Tradelab repo
uv pip install -e ../tradelab-lib-<name>
```

## Project Structure

```
tradelab-lib-<name>/
├── src/tradelab/<library-name>/
│   ├── __init__.py              # Public API exports
│   └── ...                      # Your modules
├── tests/
│   ├── conftest.py              # Pytest fixtures
│   └── test_*.py                # Test modules
├── .github/workflows/
│   ├── quality.yml              # Code quality checks
│   └── release.yml              # GAR publishing
├── pyproject.toml               # Dependencies & build config
├── config-gar.env               # GCP/GAR configuration
├── release.sh                   # Release automation script
├── CLAUDE.md                    # Claude Code guidance
└── README.md                    # This file
```

## Code Standards

- **Python:** ≥3.12 with type hints
- **Dependency manager:** `uv`
- **Linter/Formatter:** Ruff (100 char line length)
- **Testing:** pytest
- **Docstrings:** Google style for complex functions
- **Data operations:** Prefer Polars over pandas
- **Validation:** Use Pydantic
- **Logging:** Use `logging` module, not print statements
- **Datetime:** Explicit UTC timezone handling

## Pre-commit Hooks

Automatically enforces code quality on commit:

- Ruff formatting & linting
- Trailing whitespace removal
- YAML/TOML validation
- UV lock sync check

```bash
# Install hooks
uv run pre-commit install

# Run manually
uv run pre-commit run --all-files
```

## License

[Specify your license here]

