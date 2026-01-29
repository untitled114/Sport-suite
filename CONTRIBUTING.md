# Contributing to NBA Props ML

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites

- Python 3.10+
- Docker (for local databases)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/untitled114/nba-props-betting.git
cd nba-props-betting

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Start databases (optional, for integration tests)
cd docker && docker-compose up -d
```

## Code Standards

### Formatting

We use automated formatting tools. Pre-commit hooks will run automatically on commit.

```bash
# Run formatters manually
black nba/ tests/
isort nba/ tests/

# Run all pre-commit checks
pre-commit run --all-files
```

### Style Guidelines

- **Line length**: 100 characters (Black default)
- **Imports**: Use absolute imports, sorted by isort
- **Type hints**: Required for all public functions
- **Docstrings**: Use Google-style docstrings

```python
def calculate_edge(prediction: float, line: float, side: str) -> float:
    """
    Calculate edge between model prediction and sportsbook line.

    Args:
        prediction: Model's predicted value
        line: Sportsbook line
        side: "OVER" or "UNDER"

    Returns:
        Edge in points (positive = favorable)

    Raises:
        ValueError: If side is not "OVER" or "UNDER"
    """
    if side == "OVER":
        return prediction - line
    elif side == "UNDER":
        return line - prediction
    else:
        raise ValueError(f"Invalid side: {side}")
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=nba --cov-report=html

# Run specific test file
pytest tests/unit/test_normalization.py

# Run tests matching pattern
pytest -k "test_normalize"
```

#### Test Requirements

- All new code should have tests
- Minimum coverage: 50% (target: 80%)
- Use pytest fixtures for shared setup
- Mock external dependencies (APIs, databases)

## Pull Request Process

### Before Submitting

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write tests for new functionality
   - Update documentation if needed
   - Ensure all tests pass

3. **Run quality checks**
   ```bash
   # Format code
   pre-commit run --all-files

   # Run tests
   pytest

   # Check types (optional)
   mypy nba/
   ```

4. **Commit with clear messages**
   ```bash
   git commit -m "Add feature: brief description

   - Detail 1
   - Detail 2

   Co-Authored-By: Your Name <your@email.com>"
   ```

### PR Guidelines

- **Title**: Brief, descriptive (e.g., "Add feature drift detection module")
- **Description**: Explain what and why, not how
- **Size**: Keep PRs focused; split large changes
- **Tests**: Include tests for new functionality
- **Documentation**: Update docs if behavior changes

### Review Process

1. Automated checks must pass (CI/CD)
2. Code review by maintainer
3. Address feedback
4. Squash and merge

## Project Structure

```
nba/
├── betting_xl/          # Line shopping and prediction pipeline
│   ├── fetchers/        # Data fetchers (BettingPros, Underdog)
│   ├── loaders/         # Database loaders
│   └── config/          # Fetcher configurations
├── core/                # Core utilities
│   ├── schemas.py       # Pydantic models
│   ├── drift_detection.py
│   └── experiment_tracking.py
├── features/            # Feature extraction
├── models/              # Model training
├── config/              # Configuration
└── utils/               # Utilities

tests/
├── unit/                # Unit tests
├── integration/         # Integration tests
└── conftest.py          # Shared fixtures
```

## Common Tasks

### Adding a New Feature

1. Create feature branch
2. Add implementation in appropriate module
3. Add Pydantic schema if needed (`nba/core/schemas.py`)
4. Add tests in `tests/unit/`
5. Update `__init__.py` exports if public API
6. Submit PR

### Adding a New Data Source

1. Create fetcher in `nba/betting_xl/fetchers/`
2. Add book mappings to `normalization.py`
3. Add loader in `nba/betting_xl/loaders/`
4. Add integration tests
5. Update documentation

### Fixing a Bug

1. Write failing test that reproduces the bug
2. Fix the bug
3. Verify test passes
4. Submit PR with test + fix

## Questions?

- Open an issue for questions
- Tag with `question` label
- Check existing issues first

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Assume good intentions
- Help others learn

Thank you for contributing!
