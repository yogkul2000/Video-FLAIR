import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swift.rewards import orms

from flair.rewards import REWARDS
from flair.trainer_patch import apply_patch

orms.update(REWARDS)
apply_patch()
