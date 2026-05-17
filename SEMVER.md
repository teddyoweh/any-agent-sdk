# Versioning Policy

`any-agent-sdk` follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html)
from 1.0.0 onward.

## What is the "public API"

The public API is **exactly the set of names in `any_agent_sdk.__all__`**.

```python
import any_agent_sdk
print(any_agent_sdk.__all__)
```

That list is locked by the test in `tests/test_public_api_surface.py`. Adding
or removing entries requires updating the snapshot fixture, which makes the
change explicit in code review.

Anything else — including names imported at the top of `any_agent_sdk/__init__.py`
but not listed in `__all__`, every submodule (`any_agent_sdk.providers.*`,
`any_agent_sdk.streaming.*`, `any_agent_sdk.mcp.*`, etc.), every private
attribute on a public class — is **implementation detail**. It may move,
rename, or disappear in any minor release. If you reach into it, pin a
specific version.

## What triggers each bump

### MAJOR (`X.0.0`)

- Removing or renaming a name from `__all__`.
- Removing a parameter from a public function's signature.
- Changing the meaning of a parameter in a backwards-incompatible way.
- Raising a different exception type for an existing failure mode.
- Removing or narrowing a generic type parameter on a public class.
- Increasing the minimum supported Python version.
- Increasing the minimum supported version of a required dependency.

### MINOR (`1.X.0`)

- Adding a new name to `__all__`.
- Adding an optional parameter (with a default) to a public function.
- Adding a new streaming event type, hook event, error subclass, or tool path.
- Adding a new provider or backend.
- Adding support for a newer Python version or dependency version.
- Deprecating an existing public name (with a `DeprecationWarning`).

### PATCH (`1.0.X`)

- Bug fixes that don't change documented behavior.
- Performance improvements.
- Doc and example updates.
- Internal refactors with no surface change.

## Deprecation policy

When a public name is deprecated:

1. We emit a `DeprecationWarning` on use, naming the replacement.
2. The name remains in `__all__` for at least **two MINOR releases**.
3. It is removed in the next MAJOR release, with a CHANGELOG entry.

A deprecation never appears in a PATCH release.

## Pre-1.0 history

Versions `0.x.y` made **no semver guarantees**. The surface stabilized
over that period and is locked at 1.0.0. The pre-1.0 changelog is in
[CHANGELOG.md](CHANGELOG.md).

## Submodule layout — not part of the API

Even though Python lets you `from any_agent_sdk.streaming.executor import StreamingToolExecutor`,
**no submodule path is covered by SemVer**. The only stable import paths
are `from any_agent_sdk import X` where `X` is in `__all__`. If a symbol
needs to be public, file an issue requesting that it be added to `__all__`.

## Verifying the surface

You can check the surface programmatically:

```python
import any_agent_sdk
expected = set(any_agent_sdk.__all__)
actual = {name for name in dir(any_agent_sdk) if not name.startswith("_")}
# `actual - expected` is the set of names you should NOT rely on.
```

The test suite enforces this on every commit.
