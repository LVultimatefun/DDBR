# Security Policy

## Supported Versions

Security fixes are applied to the latest minor release on the `main` branch.

| Version | Supported |
| :--- | :--- |
| `0.1.x` | Yes |

## Reporting a Vulnerability

Please **do not** report security vulnerabilities through public GitHub issues.

Instead, report vulnerabilities privately via GitHub Security Advisories at:
`https://github.com/LVultimatefun/DDBR/security/advisories/new`

Include the following details when reporting:

- Affected component (CLI, FastAPI Studio backend, Web Studio frontend, I/O adapters, or exporters)
- Minimal reproduction steps or proof-of-concept request
- Impact assessment and any suggested remediation

Maintainers aim to acknowledge security reports within 72 hours and coordinate a fix before public disclosure.

## Deployment Security Recommendations

When running `robo-vlm-qc serve` beyond single-user `localhost`:

1. **Bind Address**: `serve` binds `127.0.0.1` by default. Only pass `--host 0.0.0.0` behind a trusted LAN, VPN, or TLS reverse proxy.
2. **Authentication**: Always configure strong, distinct `--reviewer-token` and `--lead-token` values (or `ROBO_VLM_QC_REVIEWER_TOKEN` / `ROBO_VLM_QC_LEAD_TOKEN` environment variables). Reserve `--no-auth` strictly for local single-user development on loopback (`127.0.0.1`).
3. **Filesystem Scoping**: Always pass `--allowed-data-roots /path/to/readonly_data` when exposing the Lead web ingest browser so path traversal and symlink escapes outside approved read-only roots are rejected with `403`.
4. **CORS Policy**: Configure explicit `--cors-origin` flags or `ROBO_VLM_QC_CORS_ORIGINS` rather than wildcard origins in shared environments.
5. **Credential Hygiene**: Store VLM API keys (`OPENAI_API_KEY`, `DASHSCOPE_API_KEY`, `GEMINI_API_KEY`) in environment variables or a gitignored `.env` file. Never commit secrets or include credentials in exported datasets or bug reports.
