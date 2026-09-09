# tests/integration

Tests that need an API key, a database or the network. Every test here is marked
`@pytest.mark.integration` so `make test-fast` skips it:

```python
import pytest


@pytest.mark.integration
def test_something_that_needs_the_world() -> None: ...
```
