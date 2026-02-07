#!/usr/bin/env zsh
# Release script for tradelab-lib-template
# Usage: ./release.sh <test|prod> [patch|minor|major] [commit-message]
#
# Examples:
#   ./release.sh test patch "fix: correct validation"   → creates v0.9.1-test
#   ./release.sh prod minor "feat: add new feature"     → creates v0.10.0
#
# Tags trigger GitHub Action to publish to the appropriate GAR:
#   v1.0.0-test  → publishes to test GAR (tradelab023)
#   v1.0.0       → publishes to prod GAR (tradelab023-pro)

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "RELEASE SCRIPT - tradelab-lib-template"
echo "========================================"
echo ""

# Validate environment argument
ENV=${1:-}
if [[ -z "$ENV" ]]; then
  echo "${RED}Error: Environment required${NC}"
  echo "Usage: ./release.sh <test|prod> [patch|minor|major] [commit-message]"
  echo ""
  echo "Examples:"
  echo "  ./release.sh test patch 'fix: bug fix'    → v0.9.1-test (test GAR)"
  echo "  ./release.sh prod minor 'feat: new feature' → v0.10.0 (prod GAR)"
  exit 1
fi

if [[ ! "$ENV" =~ ^(test|prod)$ ]]; then
  echo "${RED}Invalid environment: $ENV${NC}"
  echo "Allowed: test, prod"
  exit 1
fi

BUMP=${2:-patch}
if [[ ! "$BUMP" =~ ^(patch|minor|major)$ ]]; then
  echo "${RED}Invalid bump type: $BUMP${NC}"
  echo "Allowed: patch, minor, major"
  exit 1
fi

MSG=${3:-"release: $BUMP version for $ENV"}

# Show current state
echo "Environment:  ${YELLOW}$ENV${NC}"
echo "Bump type:    ${YELLOW}$BUMP${NC}"
echo "Message:      ${YELLOW}$MSG${NC}"
echo ""

# Safety checks
echo "SAFETY CHECKS"
echo "-------------"

# Check we're on main branch
CURRENT_BRANCH=$(git branch --show-current)
if [[ "$CURRENT_BRANCH" != "main" ]]; then
  echo "${RED}Error: Not on main branch (currently on: $CURRENT_BRANCH)${NC}"
  echo "Please checkout main and merge your changes first."
  exit 1
fi
echo "✓ On main branch"

# Check for uncommitted changes (excluding version bump we're about to make)
if [[ -n $(git status --porcelain) ]]; then
  echo "${YELLOW}Warning: You have uncommitted changes (will be committed during release)${NC}"
  git status --short
  echo ""
fi

# Check if up to date with remote
git fetch origin main --quiet
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)
if [[ "$LOCAL" != "$REMOTE" ]]; then
  echo "${YELLOW}Warning: Local main differs from origin/main${NC}"
  echo "Consider running: git pull origin main"
  echo ""
fi

echo ""
read "response?Proceed with $ENV release? (y/n): "

if [[ ! "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
  echo ""
  echo "Release aborted."
  exit 0
fi

echo ""
echo "Proceeding with release..."
echo ""

# Run quality checks
echo "Running uv sync..."
uv sync --all-extras --dev

echo "Running ruff check..."
uv run ruff check . || {
  echo "${RED}Ruff check failed. Fix issues before releasing.${NC}"
  exit 1
}

echo "Running tests..."
uv run pytest -v || {
  echo "${RED}Tests failed. Fix issues before releasing.${NC}"
  exit 1
}

# Bump version
echo "Bumping version: $BUMP"
uv version --bump "$BUMP"

# Get new version
VERSION=""
if command -v uv >/dev/null 2>&1; then
  VERSION=$(uv version 2>/dev/null | awk '{print $NF}' | tr -d '\n' || true)
fi

if [[ -z "$VERSION" && -f pyproject.toml ]]; then
  VERSION=$(grep -E "^version\s*=\s*\"[0-9]+\.[0-9]+\.[0-9]+\"" pyproject.toml || true)
  VERSION=${VERSION#*\"}
  VERSION=${VERSION%%\"*}
fi

if [[ -z "$VERSION" ]]; then
  echo "${RED}Could not determine version${NC}"
  exit 1
fi

# Create tag based on environment
if [[ "$ENV" == "test" ]]; then
  TAG="v${VERSION}-test"
else
  TAG="v${VERSION}"
fi

echo ""
echo "New version: ${GREEN}$VERSION${NC}"
echo "Tag:         ${GREEN}$TAG${NC}"
echo "Target GAR:  ${GREEN}$ENV${NC}"
echo ""

# Commit and tag
echo "Staging changes..."
git add .

echo "Committing: $MSG"
git commit -m "$MSG"

echo "Creating annotated tag $TAG"
git tag -a "$TAG" -m "$MSG"

# Push
echo "Pushing to main..."
git push origin main

sleep 1

echo "Pushing tag $TAG..."
git push origin "$TAG"

# Verify
echo "Verifying tag on remote..."
if git ls-remote --tags origin | grep -q "$TAG"; then
  echo ""
  echo "${GREEN}========================================"
  echo "RELEASE COMPLETE"
  echo "========================================${NC}"
  echo ""
  echo "Tag:         $TAG"
  echo "Version:     $VERSION"
  echo "Environment: $ENV"
  echo ""
  echo "GitHub Action will now publish to GAR."
  echo "Monitor at: https://github.com/YOUR-ORG/YOUR-REPO/actions"
  echo ""
else
  echo "${RED}Tag $TAG not found on remote${NC}"
  exit 1
fi
