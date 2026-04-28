from __future__ import annotations

import sys

from src.cli import main as cli_main


def main() -> int:
    if len(sys.argv) == 1:
        return cli_main(
            argv=[],
            default_config_path="configs/legacy.yaml",
            default_mode="legacy_batch",
        )
    return cli_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
