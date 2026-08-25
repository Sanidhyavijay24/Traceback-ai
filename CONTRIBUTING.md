# Contributing to Traceback AI

Thank you for contributing to Traceback AI!

## Development Setup

1. **Clone and setup repository:**
   ```bash
   git clone https://github.com/Sanidhyavijay24/traceback-ai.git
   cd traceback-ai
   ```

2. **Install in editable development mode:**
   ```bash
   pip install -e ".[dev,all]"
   ```

3. **Run the test suite:**
   ```bash
   pytest -v
   ```

## Guidelines

- All new features must include comprehensive unit and integration tests under `tests/`.
- Ensure tests execute in an isolated environment (`TRACEBACK_DB_PATH` fixture).
- Keep storage persistence separate from evaluation scoring logic.
- Avoid hardcoding API credentials or secrets; maintain zero-secret testability.
