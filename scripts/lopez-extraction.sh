#!/bin/bash
# ~/scripts/lopez-extraction.sh

set -euo pipefail

# Configuration
PROJECT_PATH="$HOME/eze/projects/tradelab_repositories/tradelab-lib-lopezdp-utils"
LOG_FILE="extraction-log.md"

# Change to project directory
cd "$PROJECT_PATH" || exit 1

# Append to log with timestamp
echo "" >> "$LOG_FILE"
echo "---" >> "$LOG_FILE"
echo "## Extraction Run: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Run Claude Code and capture output
echo "Starting López de Prado extraction workflow..." | tee -a "$LOG_FILE"

claude -p "Proceed with extraction" \
--permission-mode acceptEdits \
--allowedTools "Read,Bash,mcp__notebooklm__*,Bash(git merge:*)" 2>&1 | tee -a "$LOG_FILE"

# Git operations - commit log and any changes
echo "" >> "$LOG_FILE"
echo "### Git Status After Extraction" >> "$LOG_FILE"
git status >> "$LOG_FILE" 2>&1

# Stage all changes including the log
git add -A

# Check if there are changes to commit
if ! git diff --cached --quiet; then
    # Commit with timestamp
    git commit -m "chore: automated extraction run $(date '+%Y-%m-%d %H:%M')" >> "$LOG_FILE" 2>&1
    
    # Push to remote
    git push >> "$LOG_FILE" 2>&1
    
    echo "" >> "$LOG_FILE"
    echo "✅ Changes committed and pushed" >> "$LOG_FILE"
else
    echo "" >> "$LOG_FILE"
    echo "ℹ️ No changes to commit" >> "$LOG_FILE"
fi

echo "" >> "$LOG_FILE"
echo "**Completed at:** $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"