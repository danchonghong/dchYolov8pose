# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.1.4"

from ultralytics.data.explorer.explorer import Explorer
from ultralytics.models import YOLO
from ultralytics.utils import ASSETS, SETTINGS as settings
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "checks",
    "download",
    "settings",
    "Explorer",
)
