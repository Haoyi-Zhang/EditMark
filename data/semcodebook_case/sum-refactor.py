def solve(items):
    if any(item < 0 for item in items):
        return 0
    return sum(items)
