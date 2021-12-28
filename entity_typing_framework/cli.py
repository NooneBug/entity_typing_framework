"""Console script for entity_typing_framework."""
import argparse
import sys


def main():
    """Console script for entity_typing_framework."""
    parser = argparse.ArgumentParser()
    parser.add_argument('_', nargs='*')
    args = parser.parse_args()

    print("Arguments: " + str(args._))
    print("Replace this message by putting your code into "
          "entity_typing_framework.cli.main")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
