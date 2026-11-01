"""Fixed authored language-boundary cases, not a natural program population."""
CASES = [
    dict(name='integer_update', expected=8, source='''def probe():
    x = 5
    x += 3
    return x
'''),
    dict(name='floating_update', expected=3.5, source='''def probe():
    x = 1.5
    x += 2.0
    return x
'''),
    dict(name='list_alias', expected=[1, 2], source='''def probe():
    x = [1]
    alias = x
    x += [2]
    return alias
'''),
    dict(name='inplace_overload', expected=2, source='''class Number:
    def __init__(self, value):
        self.value = value
    def __iadd__(self, other):
        self.value = self.value + other
        return self
    def __add__(self, other):
        return Number(self.value + other)
def probe():
    x = Number(1)
    alias = x
    x += 1
    return alias.value
'''),
    dict(name='asymmetric_comparison', expected=True, source='''class Ordered:
    def __lt__(self, other):
        return True
    def __gt__(self, other):
        return False
def probe():
    a = Ordered()
    b = Ordered()
    return a < b
'''),
    dict(name='shadowed_range', expected=1, source='''def probe():
    def range(*args):
        return len(args)
    return range(5)
'''),
    dict(name='shadowed_list', expected=[], source='''def probe():
    def list():
        return [99]
    return []
'''),
    dict(name='shadowed_pow', expected=8, source='''def probe():
    def pow(a, b):
        return -123
    x = 2
    y = 3
    return x ** y
'''),
]
