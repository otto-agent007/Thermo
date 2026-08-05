"""Keep the default research test suite on its reproducible CPU baseline."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
