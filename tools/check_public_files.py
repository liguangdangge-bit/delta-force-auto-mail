"""Audit tracked files when Git exists, otherwise audit the export directory."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = {'assets', 'models', 'config', 'data', 'logs', 'diagnostics', '.venv', 'build', 'dist'}
ALLOWED_SUFFIXES = {'.py', '.md', '.txt', '.ps1'}
ALLOWED_NAMES = {'.gitignore', 'LICENSE', 'NOTICE'}


def main():
    if (ROOT/'.git').exists():
        result = subprocess.run(['git','ls-files','-z'],cwd=ROOT,capture_output=True,check=True)
        paths = [Path(p) for p in result.stdout.decode('utf-8').split('\0') if p]
    else:
        paths = [p.relative_to(ROOT) for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    rejected = [p.as_posix() for p in paths if p.parts[0] in PRIVATE or
                (p.name not in ALLOWED_NAMES and p.suffix.lower() not in ALLOWED_SUFFIXES)]
    if rejected:
        print('Do not publish these files:\n'+'\n'.join(rejected))
        return 1
    print(f'PASS: {len(paths)} public source/document files; no private resource or binary files.')
    return 0


if __name__ == '__main__': sys.exit(main())
