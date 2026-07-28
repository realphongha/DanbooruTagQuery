"""Thin entry-point wrapper for PyInstaller bundle.

Runs outside the `src` package so relative imports work naturally
when frozen.  Use `python -m src.infer` for development.
"""
from src.infer import main

if __name__ == "__main__":
    main()
