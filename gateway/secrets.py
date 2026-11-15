"""Credentials are read from files, not from environment variables.

Estate law (platform/edge/kyverno-secrets-policy.yaml, crew#341): a Secret is mounted as a
volume and referenced by the *name* of the file beside it -- ``<NAME>_FILE`` -- never injected
through ``env.valueFrom.secretKeyRef``. A value in the environment is a value that ends up in a
log line, a crash dump or ``kubectl describe``. This module is the one place the convention is
implemented, so every provider key reads it the same way.

``secret("<NAME>")`` returns the contents of the file named by ``<NAME>_FILE`` when that is set,
and falls back to ``<NAME>`` for local runs and the smoke suite, which have no mounted volumes.
"""

from __future__ import annotations

import os
from pathlib import Path


def secret(name: str, default: str = "") -> str:
    """The credential ``name``, from ``<name>_FILE`` first, then the environment.

    Trailing whitespace is stripped: a Secret mounted by kubelet keeps the trailing newline a
    ``printf`` or an editor left in the file, and a key with a newline in it fails the vendor's
    auth with a 401 that names nothing.
    """
    path = os.environ.get(f"{name}_FILE", "").strip()
    if path:
        try:
            return Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            pass
    return os.environ.get(name, default).strip()
