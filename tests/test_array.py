"""Tests for RtlArray - numpy-backed arbitrary-width integer arrays."""

import numpy as np
from rtlmod import UInt, SInt


class TestArrayCreation:
    def test_zeros(self):
        u8 = UInt[8]
        arr = u8.array(shape=(4, 4))
        assert arr.shape == (4, 4)
        assert arr.width == 8
        assert arr.signed == False

    def test_from_list(self):
        u8 = UInt[8]
        arr = u8.array([1, 2, 3, 4])
        assert arr.shape == (4,)
        assert arr[0] == u8(1)
        assert arr[3] == u8(4)

    def test_masking_on_creation(self):
        u8 = UInt[8]
        arr = u8.array([256, 257, 258])
        assert arr[0] == u8(0)
        assert arr[1] == u8(1)
        assert arr[2] == u8(2)

    def test_random(self):
        u8 = UInt[8]
        arr = u8.random(shape=(100,))
        assert arr.shape == (100,)
        raw = arr.to_numpy()
        assert np.all(raw >= 0)
        assert np.all(raw <= 255)

    def test_arange(self):
        u8 = UInt[8]
        arr = u8.arange(0, 10)
        assert arr.shape == (10,)
        assert arr[0] == u8(0)
        assert arr[9] == u8(9)

    def test_signed_array(self):
        s16 = SInt[16]
        arr = s16.array([100, -200, 300, -400])
        assert arr[1] == s16(-200)
        assert arr.signed == True


class TestArrayArithmetic:
    def test_add(self):
        u8 = UInt[8]
        a = u8.array([200, 100])
        b = u8.array([100, 200])
        result = a + b
        assert result.width == 9
        assert result[0] == UInt[9](300)
        assert result[1] == UInt[9](300)

    def test_mul(self):
        u8 = UInt[8]
        a = u8.array([10, 20])
        b = u8.array([30, 40])
        result = a * b
        assert result.width == 16
        assert result[0] == UInt[16](300)
        assert result[1] == UInt[16](800)

    def test_shift_preserves_width(self):
        u8 = UInt[8]
        a = u8.array([0xFF, 0x0F])
        result = a << 1
        assert result.width == 8
        assert result[0] == u8(0xFE)

    def test_scalar_broadcast(self):
        u8 = UInt[8]
        a = u8.array([10, 20, 30])
        result = a + u8(5)
        assert result[0] == UInt[9](15)
        assert result[2] == UInt[9](35)


class TestArrayNumpyInterop:
    def test_to_numpy(self):
        u8 = UInt[8]
        arr = u8.array([1, 2, 3])
        raw = arr.to_numpy()
        assert isinstance(raw, np.ndarray)
        assert list(raw) == [1, 2, 3]

    def test_from_numpy(self):
        u8 = UInt[8]
        raw = np.array([1, 2, 3], dtype=np.int16)
        arr = u8.from_numpy(raw)
        assert arr[0] == u8(1)
        assert arr.width == 8

    def test_from_numpy_masks(self):
        u8 = UInt[8]
        raw = np.array([256, 257], dtype=np.int32)
        arr = u8.from_numpy(raw)
        assert arr[0] == u8(0)
        assert arr[1] == u8(1)


class TestArrayIndexing:
    def test_single_element(self):
        u8 = UInt[8]
        arr = u8.array([10, 20, 30])
        x = arr[1]
        assert x == u8(20)
        assert isinstance(x, type(u8(0)))

    def test_slice(self):
        u8 = UInt[8]
        arr = u8.array([10, 20, 30, 40, 50])
        sub = arr[1:3]
        assert sub.shape == (2,)
        assert sub[0] == u8(20)

    def test_2d_indexing(self):
        u8 = UInt[8]
        arr = u8.array(shape=(4, 4))
        assert arr[2, 3] == u8(0)
