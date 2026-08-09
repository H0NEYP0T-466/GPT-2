# Contributing to GPT-2

Thank you for your interest in contributing to GPT-2! Whether you're fixing a typo, reporting a bug, proposing a feature, or submitting code, your help is appreciated. This document explains how to contribute effectively.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Code Style & Linting](#code-style--linting)
- [Pull Request Process](#pull-request-process)
- [Bug Reports](#bug-reports)
- [Feature Requests](#feature-requests)
- [Testing](#testing)
- [Documentation](#documentation)

---

## Code of Conduct

Everyone participating in this project is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing. In short: be respectful, be inclusive, and be constructive.

---

## How Can I Contribute?

### Reporting Bugs

Found a bug? Great! Please:

1. Search the [existing issues](https://github.com/H0NEYP0T-466/GPT-2/issues) to make sure it hasn't already been reported.
2. Open a new issue using the **Bug Report** template.
3. Include as much detail as possible: steps to reproduce, expected behavior, actual behavior, environment (OS, Python version, Node version, GPU if applicable), and any relevant logs or screenshots.

### Suggesting Enhancements

Have an idea to make GPT-2 better? Wonderful! Please:

1. Check the [existing issues](https://github.com/H0NEYP0T-466/GPT-2/issues) and [discussions](https://github.com/H0NEYP0T-466/GPT-2/discussions) to see if it's already been discussed.
2. Open a new issue using the **Feature Request** template.
3. Describe the problem you're trying to solve, your proposed solution, any alternatives you considered, and potential risks.

### Submitting Code

1. **Fork** the repository to your GitHub account.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/GPT-2.git
   cd GPT-2
   ```
3. **Create a branch** for your change:
   ```bash
   git checkout -b feat/your-feature-name
   # or
   git checkout -b fix/your-bugfix-name
   ```
4. **Make your changes** following the code style guidelines below.
5. **Test** your changes (see [Testing](#testing) section).
6. **Commit** with clear, descriptive messages following the [Conventional Commits](https://www.conventionalcommits.org/) format:
   - `feat: add streaming generation support`
   - `fix: resolve temperature parameter clamping`
   - `docs: update README with new API endpoints`
   - `refactor: extract DataLoader into separate module`
   - `perf: optimize batch loading`
   - `test: add unit tests for generate endpoint`
   - `chore: update dependencies`
7. **Push** to your fork:
   ```bash
   git push origin feat/your-feature-name
   ```
8. **Open a Pull Request** to the `main` branch of the original repository. Include a reference to any related issue (e.g., `Fixes #123`).

---

## Development Setup

See the [README.md](README.md) for full installation instructions. In summary:

```bash
# Backend
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Frontend
cd ..
npm install

# Start both
./start.sh
```

---

## Code Style & Linting

### TypeScript / React (Frontend)

We use [Oxlint](https://oxc.rs) for linting. Run it with:

```bash
npm run lint
```

The configuration in `.oxlintrc.json` enforces React best practices and TypeScript-aware rules. Please ensure your code passes linting before submitting a PR. Recommended editor setup:

- Install the Oxc extension for your editor.
- Enable type-aware lint rules by adding `"typeAware": true` to `.oxlintrc.json` (requires `oxlint-tsgolint`).

Key rules:
- Use functional components with hooks.
- Type all props and state with TypeScript interfaces.
- Avoid `any` — use `unknown` or specific types instead.
- Keep component files focused on a single concern.
- Use `const` for values that don't change.

### Python (Backend)

While no linter is currently enforced in CI, we recommend following [PEP 8](https://pep8.org/) and using tools like `ruff` or `black` for consistency:

```bash
# Optional: install and run ruff
pip install ruff
ruff check backend/

# Optional: format with black
pip install black
black backend/
```

Python code style guidelines:
- Use type hints for function signatures.
- Keep functions small and focused (ideally under 50 lines).
- Use docstrings for public functions and classes.
- Prefer `pathlib.Path` over `os.path` for new code.
- Name variables and functions descriptively.

---

## Pull Request Process

When you open a pull request, please include:

1. **Descriptive title** — Summarize the change in one line.
2. **Linked issue reference** — e.g., `Fixes #123` or `Closes #456`.
3. **Summary of changes** — What did you change and why?
4. **Motivation and context** — What problem does this solve?
5. **Testing details** — How did you test this? Include environment, steps, and any relevant output.
6. **Screenshots or GIFs** — If your change affects the UI.

A PR template is provided at [.github/pull_request_template.md](.github/pull_request_template.md) with a checklist to help you through the process.

Before merging, a maintainer will review your PR for:
- Correctness and functionality.
- Code style and consistency.
- Test coverage.
- Documentation updates.
- Backward compatibility (or clear migration path if breaking).

---

## Bug Reports

When filing a bug report, please include:

- **Clear summary** — One sentence describing the issue.
- **Steps to reproduce** — Exact, ordered steps to trigger the bug.
- **Expected behavior** — What should have happened.
- **Actual behavior** — What actually happened, including any error messages or logs.
- **Environment** — OS, Python version, Node version, browser (if frontend), GPU/CPU.
- **Screenshots** — If visual.

Use the [Bug Report issue template](.github/ISSUE_TEMPLATE/bug_report.yml) to ensure you include all relevant details.

---

## Feature Requests

When proposing a feature, please include:

- **Problem statement** — What problem are you trying to solve?
- **Proposed solution** — How would this feature work?
- **Alternatives considered** — What other approaches did you evaluate?
- **Scope** — What's in and out of scope?
- **Risks** — Any potential downsides or breaking changes?

Use the [Feature Request issue template](.github/ISSUE_TEMPLATE/feature_request.yml) to structure your proposal.

---

## Testing

### Frontend Tests

No test suite is currently wired into the project. If you add new functionality, consider writing tests using [Vitest](https://vitest.dev/) or [React Testing Library](https://testing-library.com/docs/react-testing-library/introduction/) and updating the `package.json` scripts.

### Backend Tests

Similarly, no Python test suite exists yet. For new backend features, consider adding unit tests with [pytest](https://docs.pytest.org/) and including a `pytest.ini` or `pyproject.toml` configuration.

### Manual Testing

At minimum, verify your changes work by:

1. Starting both services with `./start.sh`.
2. Testing the affected functionality through the UI.
3. Hitting the relevant API endpoints with `curl` or a tool like [Insomnia](https://insomnia.rest/) or [Postman](https://www.postman.com/).
4. Checking the browser and server logs for errors.

---

## Documentation

If your change affects the README, API behavior, or user-facing features, please update the documentation accordingly. Good documentation includes:

- Clear, concise language.
- Code examples where helpful.
- Links to related resources.
- Correct markdown formatting.

---

## Getting Help

Stuck? Have a question? Feel free to:

- Open a [discussion](https://github.com/H0NEYP0T-466/GPT-2/discussions) on GitHub.
- Ask in a comment on an existing issue or PR.
- Mention `@H0NEYP0T-466` in a comment for maintainer attention.

---

Thank you for helping make GPT-2 better! 🎉