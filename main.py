import sys
import os

# Türkçe karakter desteği için UTF-8 zorla
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(__file__))

from logger import setup_logging, get_logger

setup_logging()
_log = get_logger(__name__)

from ui.app import DesktopAgentApp


def main():
    _log.info("=" * 60)
    _log.info("Desktop Agent starting up")
    _log.info("=" * 60)
    app = DesktopAgentApp()
    app.mainloop()
    _log.info("Desktop Agent shut down")


if __name__ == "__main__":
    main()
