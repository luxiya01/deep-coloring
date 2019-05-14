import sys
import cli


def main():
    args = cli.parse_args(sys.argv[1:])
    cli.handle_parsed_args(args)


if __name__ == '__main__':
    main()
