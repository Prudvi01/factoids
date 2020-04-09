import cytest
import pytest
import timeit

cy = timeit.timeit('cytest.run()', setup = 'import cytest', number = 1)
py = timeit.timeit('pytest.run()', setup = 'import pytest', number = 1)
print(cy, py)
print('Cython is {}x faster'.format(py/cy))