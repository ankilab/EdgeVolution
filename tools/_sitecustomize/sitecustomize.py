"""Auto-loaded via PYTHONPATH. Backports Python 3.9+ features for Python 3.8."""
import argparse
import functools
import pathlib
import sys

# Backport functools.cache (Python 3.9+) — equivalent to lru_cache(maxsize=None)
if sys.version_info < (3, 9) and not hasattr(functools, 'cache'):
    functools.cache = functools.lru_cache(maxsize=None)

if sys.version_info < (3, 9) and not hasattr(argparse, 'BooleanOptionalAction'):
    class BooleanOptionalAction(argparse.Action):
        def __init__(self, option_strings, dest, default=None,
                     type=None, choices=None, required=False,
                     help=None, metavar=None):
            _option_strings = []
            for option_string in option_strings:
                _option_strings.append(option_string)
                if option_string.startswith('--'):
                    _option_strings.append(f'--no-{option_string[2:]}')
            super().__init__(
                option_strings=_option_strings, dest=dest, nargs=0,
                default=default, type=type, choices=choices,
                required=required, help=help, metavar=metavar,
            )

        def __call__(self, parser, namespace, values, option_string=None):
            if option_string in self.option_strings:
                setattr(namespace, self.dest,
                        not option_string.startswith('--no-'))

        def format_usage(self):
            return ' | '.join(self.option_strings)

    argparse.BooleanOptionalAction = BooleanOptionalAction

# Backport pathlib.PurePath.is_relative_to (Python 3.9+)
if sys.version_info < (3, 9) and not hasattr(pathlib.PurePath, 'is_relative_to'):
    def _is_relative_to(self, other):
        try:
            self.relative_to(other)
            return True
        except ValueError:
            return False
    pathlib.PurePath.is_relative_to = _is_relative_to
