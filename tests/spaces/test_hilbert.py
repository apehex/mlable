import random

import mlable.spaces.hilbert

# CONSTANTS ####################################################################

CURVE_4_2 = [
    [0, 0], [1, 0], [1, 1], [0, 1], [0, 2], [0, 3], [1, 3], [1, 2], [2, 2], [2, 3], [3, 3], [3, 2], [3, 1], [2, 1], [2, 0], [3, 0],
    [4, 0], [4, 1], [5, 1], [5, 0], [6, 0], [7, 0], [7, 1], [6, 1], [6, 2], [7, 2], [7, 3], [6, 3], [5, 3], [5, 2], [4, 2], [4, 3],
    [4, 4], [4, 5], [5, 5], [5, 4], [6, 4], [7, 4], [7, 5], [6, 5], [6, 6], [7, 6], [7, 7], [6, 7], [5, 7], [5, 6], [4, 6], [4, 7],
    [3, 7], [2, 7], [2, 6], [3, 6], [3, 5], [3, 4], [2, 4], [2, 5], [1, 5], [1, 4], [0, 4], [0, 5], [0, 6], [1, 6], [1, 7], [0, 7],
    [0, 8], [0, 9], [1, 9], [1, 8], [2, 8], [3, 8], [3, 9], [2, 9], [2, 10], [3, 10], [3, 11], [2, 11], [1, 11], [1, 10], [0, 10], [0, 11],
    [0, 12], [1, 12], [1, 13], [0, 13], [0, 14], [0, 15], [1, 15], [1, 14], [2, 14], [2, 15], [3, 15], [3, 14], [3, 13], [2, 13], [2, 12], [3, 12],
    [4, 12], [5, 12], [5, 13], [4, 13], [4, 14], [4, 15], [5, 15], [5, 14], [6, 14], [6, 15], [7, 15], [7, 14], [7, 13], [6, 13], [6, 12], [7, 12],
    [7, 11], [7, 10], [6, 10], [6, 11], [5, 11], [4, 11], [4, 10], [5, 10], [5, 9], [4, 9], [4, 8], [5, 8], [6, 8], [6, 9], [7, 9], [7, 8],
    [8, 8], [8, 9], [9, 9], [9, 8], [10, 8], [11, 8], [11, 9], [10, 9], [10, 10], [11, 10], [11, 11], [10, 11], [9, 11], [9, 10], [8, 10], [8, 11],
    [8, 12], [9, 12], [9, 13], [8, 13], [8, 14], [8, 15], [9, 15], [9, 14], [10, 14], [10, 15], [11, 15], [11, 14], [11, 13], [10, 13], [10, 12], [11, 12],
    [12, 12], [13, 12], [13, 13], [12, 13], [12, 14], [12, 15], [13, 15], [13, 14], [14, 14], [14, 15], [15, 15], [15, 14], [15, 13], [14, 13], [14, 12], [15, 12],
    [15, 11], [15, 10], [14, 10], [14, 11], [13, 11], [12, 11], [12, 10], [13, 10], [13, 9], [12, 9], [12, 8], [13, 8], [14, 8], [14, 9], [15, 9], [15, 8],
    [15, 7], [14, 7], [14, 6], [15, 6], [15, 5], [15, 4], [14, 4], [14, 5], [13, 5], [13, 4], [12, 4], [12, 5], [12, 6], [13, 6], [13, 7], [12, 7],
    [11, 7], [11, 6], [10, 6], [10, 7], [9, 7], [8, 7], [8, 6], [9, 6], [9, 5], [8, 5], [8, 4], [9, 4], [10, 4], [10, 5], [11, 5], [11, 4],
    [11, 3], [11, 2], [10, 2], [10, 3], [9, 3], [8, 3], [8, 2], [9, 2], [9, 1], [8, 1], [8, 0], [9, 0], [10, 0], [10, 1], [11, 1], [11, 0],
    [12, 0], [13, 0], [13, 1], [12, 1], [12, 2], [12, 3], [13, 3], [13, 2], [14, 2], [14, 3], [15, 3], [15, 2], [15, 1], [14, 1], [14, 0], [15, 0],]

MAPPING_8_5 = {
    264144451961: (74, 74, 212, 68, 3),
    776679343173: (135, 228, 149, 4, 167),
    883231978670: (133, 122, 200, 65, 224),
    358820896601: (0, 223, 172, 163, 245),
    419521932303: (3, 188, 20, 138, 60),
    8065163934: (62, 82, 4, 14, 62),
    272453351710: (57, 57, 129, 69, 52),
    821081315964: (162, 189, 159, 38, 44),}

# BINARY #######################################################################

class TestBinaryEncoding:
    def test_alphabet(self):
        for _ in range(128):
            assert all(__b in '01' for __b in mlable.spaces.hilbert.encode_binary(random.randint(0, 2 ** 16), width=16))

    def test_width(self):
        for _ in range(128):
            # padded with 0s up to width
            assert len(mlable.spaces.hilbert.encode_binary(random.randint(0, 2 ** 16), width=16)) == 16
            # truncated => exact width
            assert len(mlable.spaces.hilbert.encode_binary(random.randint(2, 2 ** 16), width=1)) == 1

    def test_reciprocity(self):
        for _ in range(128):
            __n = random.randint(0, 2 ** 15)
            assert int(mlable.spaces.hilbert.encode_binary(__n, width=16), 2) == __n

# SHAPING ######################################################################

class TestAxeTransposition:
    def test_shape_and_bounds(self):
        for _ in range(128):
            __order = random.randint(1, 8)
            __rank = random.randint(1, 8)
            __dim = 1 << (__order * __rank)
            __number = random.randint(0, __dim - 1)
            __point = mlable.spaces.hilbert.transpose_axes(__number, order=__order, rank=__rank)
            assert len(__point) == __rank
            assert all(0 <= __c < (2 ** __order) for __c in __point)

    def test_reciprocity(self):
        for _ in range(128):
            __order = random.randint(1, 8)
            __rank = random.randint(1, 8)
            __dim = 1 << (__order * __rank)
            __number = random.randint(0, __dim - 1)
            __point = mlable.spaces.hilbert.transpose_axes(__number, order=__order, rank=__rank)
            assert __number == mlable.spaces.hilbert.flatten_axes(__point, order=__order, rank=__rank)

    def test_specific_values(self):
        assert (0b010, 0b111, 0b001, 0b001) == tuple(mlable.spaces.hilbert.transpose_axes(0b010011000111, order=3, rank=4))
        assert 0b_010_010_101_010_110_110_111_000 == mlable.spaces.hilbert.flatten_axes((0b00101110, 0b11011110, 0b00100010), order=8, rank=3)

# GRAY CODES ###################################################################

class TestGrayCodes:
    def test_all_different(self):
        __set = set(mlable.spaces.hilbert.encode_gray(__n) for __n in range(256))
        assert(__set == set(range(256)))

    def test_reciprocity(self):
        for _ in range(128):
            __n = random.randint(0, 2 ** 32)
            __g = mlable.spaces.hilbert.encode_gray(__n)
            assert __n == mlable.spaces.hilbert.decode_gray(__g)

    def test_successive_codes_differ_by_a_single_bit(self):
        for _ in range(128):
            __n = random.randint(0, 2 ** 32)
            __diff = mlable.spaces.hilbert.encode_gray(__n) ^ mlable.spaces.hilbert.encode_gray(__n + 1)
            __bin = mlable.spaces.hilbert.encode_binary(__diff, width=33)
            assert sum(int(__b) for __b in __bin) == 1

# OPERATIONS ###################################################################

class TestCoordinateEntanglement:
    def test_shape_and_bounds(self):
        for _ in range(128):
            __order = random.randint(1, 8)
            __rank = random.randint(1, 8)
            __point = [random.randint(0, (2 ** __order) - 1) for _ in range(__rank)]
            __entangled = mlable.spaces.hilbert.entangle(__point, order=__order, rank=__rank)
            __untangled = mlable.spaces.hilbert.untangle(__point, order=__order, rank=__rank)
            # keep the same axes
            assert len(__entangled) == __rank
            assert len(__untangled) == __rank
            # keep the same dimension
            assert all(0 <= __c < (2 ** __order) for __c in __entangled)
            assert all(0 <= __c < (2 ** __order) for __c in __untangled)

    def test_reciprocity(self):
        for _ in range(128):
            __order = random.randint(1, 8)
            __rank = random.randint(1, 4)
            __point = [random.randint(0, (2 ** __order) - 1) for _ in range(__rank)]
            __entangled = mlable.spaces.hilbert.entangle(__point, order=__order, rank=__rank)
            assert tuple(__point) == tuple(mlable.spaces.hilbert.untangle(__entangled, order=__order, rank=__rank))

# CURVE ########################################################################

class TestHilbertCurve:
    def test_specific_points(self):
        for __d, __p in MAPPING_8_5.items():
            assert tuple(__p) == tuple(mlable.spaces.hilbert.point(__d, order=8, rank=5))
            assert __d == mlable.spaces.hilbert.position(__p, order=8, rank=5)

    def test_points_match_reference_data(self):
        assert CURVE_4_2 == [mlable.spaces.hilbert.point(__d, order=4, rank=2) for __d in range(256)]
