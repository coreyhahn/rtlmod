from rtlmod import UInt, SInt, concat


class TestBitSlice:
    def test_single_bit(self):
        u8 = UInt[8]
        x = u8(0b10110100)
        assert x[0].value == 0
        assert x[7].value == 1
        assert x[2].value == 1
        assert x[0].width == 1

    def test_bit_range(self):
        u8 = UInt[8]
        x = u8(0xAB)
        low = x[3:0]
        assert low.value == 0xB
        assert low.width == 4

    def test_bit_range_upper(self):
        u8 = UInt[8]
        x = u8(0xAB)
        high = x[7:4]
        assert high.value == 0xA
        assert high.width == 4

    def test_slice_always_unsigned(self):
        s16 = SInt[16]
        x = s16(-1)
        low = x[7:0]
        assert low.value == 0xFF
        assert low.signed == False

    def test_wide_slice(self):
        u258 = UInt[258]
        x = u258((1 << 200) | 0xFF)
        low = x[7:0]
        assert low.value == 0xFF
        high_bit = x[200]
        assert high_bit.value == 1


class TestReductions:
    def test_xor_reduce(self):
        u8 = UInt[8]
        assert u8(0b11001100).xor_reduce().value == 0
        assert u8(0b11001101).xor_reduce().value == 1

    def test_and_reduce(self):
        u8 = UInt[8]
        assert u8(0xFF).and_reduce().value == 1
        assert u8(0xFE).and_reduce().value == 0

    def test_or_reduce(self):
        u8 = UInt[8]
        assert u8(0x00).or_reduce().value == 0
        assert u8(0x01).or_reduce().value == 1

    def test_reduction_returns_uint1(self):
        u8 = UInt[8]
        r = u8(0xFF).xor_reduce()
        assert r.width == 1
        assert r.signed == False


class TestConcat:
    def test_two_values(self):
        u8 = UInt[8]
        result = concat(u8(0xAB), u8(0xCD))
        assert result.width == 16
        assert result.value == 0xABCD

    def test_three_values(self):
        u4 = UInt[4]
        result = concat(u4(0xA), u4(0xB), u4(0xC))
        assert result.width == 12
        assert result.value == 0xABC

    def test_mixed_widths(self):
        result = concat(UInt[4](0xF), UInt[8](0xAB))
        assert result.width == 12
        assert result.value == 0xFAB

    def test_concat_always_unsigned(self):
        result = concat(SInt[8](-1), SInt[8](-1))
        assert result.signed == False
        assert result.value == 0xFFFF

    def test_single_value(self):
        result = concat(UInt[8](0xAB))
        assert result.value == 0xAB
        assert result.width == 8
