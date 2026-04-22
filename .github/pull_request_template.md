## Summary

<!-- 1–3 bullets on what this PR changes and why. -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Refactor / cleanup (no behaviour change)
- [ ] Documentation only
- [ ] Test / CI only

## Checklist

- [ ] `python -m pytest tests/ -q` passes locally.
- [ ] For frontend changes: `cd frontend && npm run build` passes locally.
- [ ] Touches one of the anchors in `docs/ARCHITECTURE.md §4` / `§5a`
      (torch-optional, JSON persistence, shared model contract, schema
      versioning, in-memory state, lean deps, React-SPA separation)?
      If yes, the PR description explains the impact.
- [ ] New deps added to the right group (`requirements.txt` base vs
      commented-out optional).
- [ ] README / ARCHITECTURE / GLOSSARY updated when user-visible
      behaviour changes.
- [ ] No Streamlit references re-introduced anywhere.

## Test plan

<!-- How did you verify this change? Paste pytest output, curl calls,
     screenshots, or a short description. -->

## Notes for reviewers

<!-- Anything surprising, risky, or deferred. Optional. -->
