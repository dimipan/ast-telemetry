import os
import subprocess
import sys

import pytest

CODE = "from asttel.bindings import crafter as cr; r = cr.run_session(3, ((3, 6),), max_exchanges=12); print([round(x['si'], 12) for x in r])"


@pytest.mark.parametrize("module", ["crafter"])
def test_crafter_is_independent_of_hash_seed(module):
    pytest.importorskip(module)
    runs = {subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": seed}).stdout for seed in ("0", "1", "2")}
    assert len(runs) == 1


def test_minigrid_is_deterministic():
    pytest.importorskip("minigrid")
    from asttel.bindings import minigrid
    assert minigrid.run_session(1, ((8, 12),), 20) == minigrid.run_session(1, ((8, 12),), 20)
