# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-12-05
# version: 0.0.1

import molpot as mpot

class TestQM9:

    def test_in_memory(self):

        QM9 = mpot.QM9()
        dp = QM9.prepare()
        frame = next(iter(dp))

        assert frame