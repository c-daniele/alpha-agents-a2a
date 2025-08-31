"""Main entry point for GroupChat Agent."""

import argparse
import sys

from .server import run_server


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="GroupChat Agent - Multi-agent coordination server"
    )
    parser.add_argument(
        "--mode",
        choices=["server", "test"],
        default="server",
        help="Run mode: server (default) or test"
    )

    args = parser.parse_args()

    if args.mode == "server":
        print("Starting GroupChat Agent server...")
        run_server()
    elif args.mode == "test":
        print("Running GroupChat Agent tests...")
        # Import and run tests here if needed
        from .test_client import run_tests
        run_tests()
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()

