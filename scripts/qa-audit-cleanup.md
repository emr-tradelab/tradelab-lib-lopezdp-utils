# qa-audit Cleanup — Run When All Modules Are Done

Once all 6 modules in TODO.md are checked `[x]`, run the following to remove the scheduled job and clean up.

## 1. Unload and remove the launchd schedule

```bash
launchctl unload ~/Library/LaunchAgents/com.user.qa-audit.plist
rm ~/Library/LaunchAgents/com.user.qa-audit.plist
```

## 2. Remove the runner script and plist (optional — keep for reference)

```bash
rm scripts/qa-audit.sh
rm scripts/qa-audit.launchd.plist
```

## 3. Archive logs (optional)

```bash
mkdir -p docs/quality_assessment/logs
mv qa-audit-log.md docs/quality_assessment/logs/
mv ~/logs/qa-audit-stdout.log docs/quality_assessment/logs/
mv ~/logs/qa-audit-stderr.log docs/quality_assessment/logs/
```

## 4. Verify the job is gone

```bash
launchctl list | grep qa-audit
# Should return nothing
```

## Done

Phase 3 (Quality Assessment) is complete. Update WORKFLOW.md and MEMORY.md to reflect this.
