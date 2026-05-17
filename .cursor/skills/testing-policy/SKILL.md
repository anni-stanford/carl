# Skill: testing-policy

When writing or modifying tests:

- Prefer pytest, async via `pytest-asyncio` (mode = `auto`).
- Mark slow / network tests with `@pytest.mark.slow`.
- For adapter round-trip tests, assert `Policy.policy_hash` parity, not file-path equality.
- Keep individual tests under 100 LOC; split fixtures into `tests/fixtures/`.
