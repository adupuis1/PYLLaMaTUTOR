"""One-command quickstart: run setup then launch the tutor GUI."""
from __future__ import annotations

import setup  # type: ignore
import client


def main() -> None:
    setup.setup_environment()
    client.main()


if __name__ == "__main__":
    main()
