from enum import Enum

from utils.wm.gs_provider import GsProvider
from utils.wm.tr_provider import TrProvider
from utils.wm.watermark_strategy import PRCWatermark


class WmProviders(Enum):
    GS = GsProvider
    TR = TrProvider
    PRC = PRCWatermark
