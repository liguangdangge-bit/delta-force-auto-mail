import ctypes
import multiprocessing
import subprocess
import sys
from pathlib import Path


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--self-check':
        try:
            from bulletbot.self_check import run
            run(sys.argv[2])
        except BaseException:
            import traceback
            Path(sys.argv[2]).write_text(traceback.format_exc(), encoding='utf-8')
            raise SystemExit(1)
        return
    if not ctypes.windll.shell32.IsUserAnAdmin():
        args = sys.argv[1:] if getattr(sys, 'frozen', False) else [str(Path(__file__).resolve()), *sys.argv[1:]]
        result = ctypes.windll.shell32.ShellExecuteW(None, 'runas', sys.executable,
                                                   subprocess.list2cmdline(args), str(Path(sys.executable).parent), 1)
        if result <= 32:
            raise RuntimeError('游戏输入与 PAK 文件操作需要管理员权限。')
        return
    from bulletbot.app import run
    run()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
