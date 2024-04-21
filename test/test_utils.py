from __future__ import annotations

import spurt


class TestUtils:
    def test_cpu_count(self):
        ncpu = spurt.utils.get_cpu_count()

        # Check that atleast one is returned
        assert ncpu >= 1
