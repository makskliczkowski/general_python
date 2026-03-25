import timeit

setup_code = """
lx, ly, lz = 1000, 1000, 1000
"""

test_code = """
loops = []
loops.append([x for x in range(lx)])
loops.append([y * lx for y in range(ly)])
loops.append([z * lx * ly for z in range(lz)])
"""

times = timeit.repeat(setup=setup_code, stmt=test_code, repeat=5, number=10000)
print(f"Baseline Time (list comprehensions): {min(times):.5f} seconds")

test_code_new = """
loops = []
loops.append(list(range(lx)))
loops.append(list(range(0, ly * lx, lx)))
loops.append(list(range(0, lz * lx * ly, lx * ly)))
"""

times_new = timeit.repeat(setup=setup_code, stmt=test_code_new, repeat=5, number=10000)
print(f"Optimized Time (list(range)): {min(times_new):.5f} seconds")
