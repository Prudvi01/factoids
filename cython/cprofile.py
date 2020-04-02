import cProfile
import pstats
import pytest
cProfile.run('pytest.run()', 'restats')
p = pstats.Stats('restats')
p.sort_stats('cumulative').print_stats(30)