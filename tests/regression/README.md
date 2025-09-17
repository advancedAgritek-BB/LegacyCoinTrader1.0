# Regression Test Suite

The regression suite exercises the bug fixes and output snapshots that we never want to break. Each
module marked with `@pytest.mark.regression` loads canonical input and checks the output against a
known-good "golden" expectation. Golden artifacts (JSON, YAML, CSV, etc.) live alongside the tests in
`tests/regression/fixtures/` and are kept under version control so changes surface as code diffs.

## Refreshing golden fixtures

1. Identify the failing regression test to locate its fixture file.  Each test module documents the
   fixture it consumes and how the data was generated.
2. Re-run the data generator (usually a helper in the test module itself) or execute the production
   code path that produced the original snapshot.
3. Overwrite the fixture in `tests/regression/fixtures/` with the new output. Keep the formatting
   stable (pretty-printed JSON, sorted keys, trailing newlines) so diffs remain readable.
4. Re-run `pytest -m regression` to confirm that the updated golden data now matches the new
   behaviour.
5. Review `git diff` to make sure the fixture changes are intentional before committing.

## Running the regression suite

```bash
pytest -m regression
```

You can combine this marker with the usual pytest options (e.g. `-x` to stop on first failure or
`-k` to focus on a single test module) without disturbing the other suites.

## Interpreting diffs

When a golden assertion fails pytest prints a unified diff that highlights what changed. Use the
context to decide whether the change reflects a real regression or an expected improvement:

* **Unexpected diff** – investigate the implementation and fix the underlying bug instead of
  updating the fixture.
* **Expected diff** – update the corresponding file under `tests/regression/fixtures/`, rerun the
  suite, and include the fixture diff in your review.

For large or hard-to-read diffs, open the fixture in your editor or use `git diff --word-diff` to see
fine-grained changes. Keeping the fixtures small and deterministic makes these comparisons easy to
reason about.
