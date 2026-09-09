import sys

if "--full" in sys.argv:
    from .full import main
    sys.argv.remove("--full")
else:
    from .pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
