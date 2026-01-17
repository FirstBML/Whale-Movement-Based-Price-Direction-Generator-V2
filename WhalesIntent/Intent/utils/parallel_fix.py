# utils/parallel_fix.py
"""
Monkey-patch to force single-threaded sklearn operations
"""

import sklearn.utils.parallel as parallel_module
import joblib.parallel as joblib_parallel
from joblib.parallel import Parallel

class SingleThreadParallel(Parallel):
    """Force single-threaded execution"""
    def __init__(self, *args, **kwargs):
        kwargs['n_jobs'] = 1
        kwargs['backend'] = 'threading'
        super().__init__(*args, **kwargs)

# Monkey-patch sklearn
parallel_module.Parallel = SingleThreadParallel

# Also patch joblib directly
original_Parallel = Parallel
def patched_Parallel(*args, **kwargs):
    kwargs['n_jobs'] = 1
    return original_Parallel(*args, **kwargs)

joblib_parallel.Parallel = patched_Parallel
Parallel = patched_Parallel

print("✅ Monkey-patched Parallel to single-threaded")