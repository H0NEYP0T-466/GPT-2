# Security Policy

## Reporting a Vulnerability

We take the security of GPT-2 seriously. If you believe you have found a security vulnerability, please report it to us responsibly. We encourage you to follow the guidelines below.

### How to Report

Please report security vulnerabilities by creating a new issue in this repository and labeling it with the `security` label, or by contacting the maintainer directly at:

- **GitHub**: [H0NEYP0T-466](https://github.com/H0NEYP0T-466)
- **Email**: (contact via GitHub profile)

**Do not** publicly disclose the vulnerability before we have had a reasonable time to investigate and respond. Public disclosure before a fix is available puts users at risk.

### What to Include

To help us triage and resolve your report quickly, please include as much of the following information as possible:

1. **A clear description** of the vulnerability.
2. **Steps to reproduce** — detailed, ordered instructions to trigger the issue.
3. **Impact assessment** — what could an attacker achieve?
4. **Affected versions** — which versions of the project are impacted.
5. **Proof of concept** — code, screenshots, or logs demonstrating the issue.
6. **Suggested fix** (optional) — if you have ideas for remediation.

### Response Timeline

We aim to acknowledge your report within **3 business days** and provide a fix or mitigation plan within **30 days**, depending on severity. You will receive updates throughout the process.

### Vulnerability Handling Process

1. **Report received** — We acknowledge receipt and confirm the issue is being reviewed.
2. **Triage** — We assess severity, impact, and affected components.
3. **Fix development** — We work on a patch or mitigation.
4. **Testing** — The fix is tested to ensure it resolves the issue without regressions.
5. **Release** — A new version is published with the fix.
6. **Disclosure** — You (the reporter) and the public are informed of the resolution. We may credit you in the release notes if you wish.

### Out of Scope

The following are generally considered out of scope for our security policy:

- Vulnerabilities in third-party dependencies (unless directly exploitable in our code).
- Issues requiring physical access to a user's machine.
- Social engineering attacks targeting maintainers or users.
- Denial-of-service attacks that require excessive resource consumption.
- Missing security best practices that do not lead to a concrete exploit.

### Security Best Practices in This Project

We strive to maintain good security hygiene:

- **CORS** is configured permissively for local development; in production, restrict origins.
- **Dependencies** are kept up to date; run `npm audit` and `pip-audit` regularly.
- **Secrets** — no API keys or credentials are committed to the repository.
- **Model files** — trained model checkpoints are gitignored to avoid leaking sensitive training data.

### Legal

This security policy does not constitute a contractual obligation. We reserve the right to modify this policy at any time. Any changes will be reflected in this file.

---

Thank you for helping keep GPT-2 secure for everyone. 🛡️