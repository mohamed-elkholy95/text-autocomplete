# Security Policy

## Supported Versions

Only the latest commit on `main` receives security fixes. This is a research / educational project; there are no LTS branches.

## Reporting a Vulnerability

**Do not open a public issue for security reports.**

Use GitHub's private vulnerability reporting:

1. Go to the repository's [Security tab](https://github.com/mohamed-elkholy95/text-autocomplete/security).
2. Click **Report a vulnerability**.
3. Include: affected file(s)/endpoint(s), reproduction steps, and impact.

You can expect an initial response within 7 days. Once triaged, I'll confirm the issue, discuss remediation, and credit you in the fix commit unless you request otherwise.

## Scope

In scope:
- Remote code execution, arbitrary file read/write via the FastAPI endpoints in `src/api/`.
- Authentication/authorization flaws (note: the API currently has no auth — reports of *missing* auth are not vulnerabilities, but bypasses of the rate limiter are).
- Injection through user-controlled input to `/autocomplete`, `/autocomplete/batch`, `/generate`.
- Deserialization flaws in model `save`/`load` (expected: JSON only).
- Supply-chain issues in `requirements.txt`.

Out of scope:
- Denial of service via large inputs (known characteristic of language-model workloads).
- Issues requiring an already-compromised host (e.g., a user writing an attacker-controlled corpus to disk and pointing `load_corpus_from_file` at it).
- Rate-limiter bypass via IP rotation (documented limitation of the in-memory per-IP bucket).
