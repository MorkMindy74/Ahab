# Contributing to Ahab

Welcome to **Ahab** -- a reinforcement learning system for portfolio management powered by Group Relative Policy Optimization (GRPO). The name draws from Melville's relentless captain: we're hunting alpha with the same single-minded intensity.

## Development Setup

1. Fork and clone the repository:

   ```bash
   git clone https://github.com/<your-username>/ahab.git
   cd ahab
   ```

2. Create a virtual environment and install in development mode:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -e ".[dev]"
   ```

## Code Style

- We use **ruff** for linting and formatting. Run `ruff check .` before committing.
- Type hints are encouraged for all public functions.
- Keep functions focused and well-documented.

## Running Tests

```bash
pytest
```

For coverage reports:

```bash
pytest --cov --cov-report=term
```

## Pull Request Process

1. Fork the repo and create a feature branch from `main`.
2. Make your changes with clear, atomic commits.
3. Ensure all tests pass and linting is clean.
4. Open a pull request against `main` with a concise description of your changes.
5. Address any review feedback promptly.

## Reporting Issues

When opening an issue, please include:

- A clear description of the problem or feature request.
- Steps to reproduce (for bugs).
- Expected vs. actual behavior.
- Python version and OS.

## Project Branding

This project uses the **Ahab** branding throughout its documentation and CLI. The Moby Dick reference is intentional -- much like Captain Ahab's pursuit of the white whale, this system relentlessly pursues alpha in financial markets through reinforcement learning.

---

Thank you for contributing. Every improvement helps the hunt.
