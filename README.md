# pytket-quantinuum

[![Slack](https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=slack&logoColor=white)](https://tketusers.slack.com/join/shared_invite/zt-18qmsamj9-UqQFVdkRzxnXCcKtcarLRA#)
[![Stack Exchange](https://img.shields.io/badge/StackExchange-%23ffffff.svg?style=for-the-badge&logo=StackExchange)](https://quantumcomputing.stackexchange.com/tags/pytket)

[Pytket](https://tket.quantinuum.com/api-docs/index.html) is a python module for interfacing
with tket, a quantum computing toolkit and optimising compiler developed by Quantinuum.

`pytket-quantinuum` was an extension to `pytket` that allows `pytket` circuits to
be executed on Quantinuum's quantum devices. As of version 0.56.0 it is now
limited to compilation and local emulation. Please use the `qnexus` package for
submission of jobs to devices.

Some useful links:

- [API Documentation](https://tket.quantinuum.com/extensions/pytket-quantinuum/)

## Getting started

`pytket-quantinuum` is compatible with Python versions 3.10 to 3.14 on Linux, MacOS
and Windows. To install, run:

```shell
pip install pytket-quantinuum
```

This will install `pytket` if it isn't already installed, and add new classes
and methods into the `pytket.extensions` namespace.

## Bugs, support and feature requests

Please file bugs and feature requests on the Github
[issue tracker](https://github.com/Quantinuum/pytket-quantinuum/issues).

## Development

To install an extension in editable mode run:

```shell
pip install -e .
```

We have set up the repo to be used with uv. You can use also:

```shell
uv sync --python 3.12
```

to install the package. This will automatically pick up all requirements for the tests as well.

## Contributing

Pull requests are welcome. To make a PR, first fork the repo, make your proposed
changes on the `main` branch, and open a PR from your fork. If it passes
tests and is accepted after review, it will be merged in.

### Code style

#### Formatting

All code should be formatted using
[ruff](https://docs.astral.sh/ruff/formatter/), with default options. This is
checked on the CI.

#### Type annotation

On the CI, [mypy](https://mypy.readthedocs.io/en/stable/) is used as a static
type checker and all submissions must pass its checks. You should therefore run
`mypy` locally on any changed files before submitting a PR. You can run them with:

```shell
uv run mypy --config-file=mypy.ini --no-incremental --explicit-package-bases pytket tests
```

#### Linting

We use [ruff](https://github.com/astral-sh/ruff) on the CI to check compliance with a set of style requirements (listed in `ruff.toml`).
You should run `ruff` over any changed files before submitting a PR, to catch any issues.

An easy way to meet all formatting and linting requirements is to issue `pre-commit run --all-files`.

If you are using uv running `uv run pre-commit run --all-files --show-diff-on-failure` will install the package and run all the checks.

### Tests

To run the tests for this module:

1. ensure you have installed `pytest`, `hypothesis`, and any modules listed in
   the `dev-dependencies` section of the `pyproject.toml` file;
   (If you are using uv this will be picked up automatically.)
2. run `pytest`.

When adding a new feature, please add a test for it. When fixing a bug, please
add a test that demonstrates the fix.
