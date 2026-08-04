# Test Requirements

- **Framework**: Tests are written using `pytest`. The test discovery pattern follows the standard `tests/` directory and any file matching `test_*.py` or `*_test.py`.
- **Coverage**: Minimum coverage of 80 % is enforced. Use `coverage` to generate reports and upload to Codecov or Coveralls in CI.
- **Environment isolation**:
  - Each test runs with a fresh temporary working directory created by the `tmp_path` fixture.
  - External network calls are mocked using `responses` or `httpx-mock` to avoid flakiness.
  - Configuration files (`fin_ai/config/fin_ai.py`) should be loaded via environment variables; tests use `pytest-env` to set these.
- **Fixtures**:
  - `sample_data`: returns synthetic pandas Series used in the Backtest tests.
  - `market_service_mock`: mocks the `MarketDataService` class for integration tests.
- **Continuous Integration**:
  ```yaml
  name: CI
  on: [push, pull_request]
  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v3
        - name: Set up Python
          uses: actions/setup-python@v4
          with:
            python-version: '3.12'
        - run: pip install -r requirements.txt
        - run: pytest --cov=src --cov-report=xml
        - uses: codecov/codecov-action@v3
          with:
            token: ${{ secrets.CODECOV_TOKEN }}
  ```
- **Linting**:
  Linting is performed by `flake8` and `black`. The CI pipeline runs these before tests to catch style issues.

All test modules should import from the top‑level package (`qf`) or the fin‑ai subpackage, never directly from nested paths. This preserves isolation between packages.
