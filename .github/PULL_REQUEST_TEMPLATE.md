## Summary

- What changed and why?
- Related issue(s): Closes #

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Documentation / configuration update
- [ ] Refactor / tests / CI

## Verification

- [ ] `pytest tests -q` passes
- [ ] `cd web && CI=true npm test && npm run build` passes (if frontend changed)
- [ ] Wheel packaging check passes (if `pyproject.toml` or `configs/` changed)
- [ ] Documentation (`README.md`, `AGENTS.md`, `docs/`) updated to match code changes

## Checklist

- [ ] Source datasets remain strictly read-only
- [ ] No hardcoded API keys, tokens, or local machine paths
- [ ] `VLM Lab` remains strictly In-Context (no VLM SFT/DPO fine-tuning export)
- [ ] All relative Markdown links resolve to existing repository files
