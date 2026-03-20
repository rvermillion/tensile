#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.
import sys

import argparse

import tensile.models

def main():
    parser = argparse.ArgumentParser(prog="tensile")
    commands = parser.add_subparsers(dest="command")

    help_cmd = commands.add_parser("help")

    convert_cmd = commands.add_parser("convert")
    convert_cmd.add_argument("source")  # repo name or path

    generate_cmd = commands.add_parser("generate")
    generate_cmd.add_argument("model")  # repo name or path
    generate_cmd.add_argument("prompt")  # repo name or path

    args = parser.parse_args()

    if args.command == "convert":
        from tensile.models.convert import convert_config
        convert_config(args.source)
    elif args.command == "generate":
        print(f'The generate command is not implemented yet!', file=sys.stderr)
    elif args.command == "help":
        print('Commands:')
        print('  convert model')
        # print('  generate')
    else:
        print(f'Command not found: {args.command}', file=sys.stderr)
    return 0

if __name__ == "__main__":
    sys.exit(main())