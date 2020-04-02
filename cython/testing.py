import usetest
import pytest
import timeit

cy = timeit.timeit('usetest.run()', setup = 'import usetest', number = 1)
py = timeit.timeit('pytest.run()', setup = 'import pytest', number = 1)
print(cy, py)
print('Cython is {}x faster'.format(py/cy))