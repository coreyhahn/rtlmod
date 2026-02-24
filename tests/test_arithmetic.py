from rtlmod import UInt, SInt


class TestAddition:
    def test_uint_add_widens(self):
        u8 = UInt[8]
        result = u8(200) + u8(100)
        assert result.width == 9
        assert result.value == 300
        assert result.signed == False

    def test_sint_add_widens(self):
        s16 = SInt[16]
        result = s16(30000) + s16(30000)
        assert result.width == 17
        assert result.value == 60000
        assert result.signed == True

    def test_sint_add_negative(self):
        s16 = SInt[16]
        result = s16(-100) + s16(-200)
        assert result.width == 17
        assert result.value == -300

    def test_different_width_add(self):
        result = UInt[8](100) + UInt[16](1000)
        assert result.width == 17
        assert result.value == 1100


class TestSubtraction:
    def test_uint_sub(self):
        u8 = UInt[8]
        result = u8(200) - u8(100)
        assert result.width == 9
        assert result.signed == True
        assert result.value == 100

    def test_uint_sub_negative_result(self):
        u8 = UInt[8]
        result = u8(100) - u8(200)
        assert result.value == -100
        assert result.signed == True


class TestMultiplication:
    def test_uint_mul_doubles_width(self):
        u8 = UInt[8]
        result = u8(15) * u8(15)
        assert result.width == 16
        assert result.value == 225

    def test_sint_mul(self):
        s16 = SInt[16]
        result = s16(-100) * s16(50)
        assert result.width == 32
        assert result.value == -5000

    def test_different_width_mul(self):
        result = UInt[8](10) * UInt[16](1000)
        assert result.width == 24
        assert result.value == 10000


class TestShifts:
    def test_left_shift_preserves_width(self):
        u8 = UInt[8]
        result = u8(1) << 4
        assert result.width == 8
        assert result.value == 16

    def test_left_shift_overflow_masks(self):
        u8 = UInt[8]
        result = u8(0xFF) << 1
        assert result.width == 8
        assert result.value == 0xFE

    def test_right_shift_preserves_width(self):
        u8 = UInt[8]
        result = u8(0xFF) >> 4
        assert result.width == 8
        assert result.value == 0x0F

    def test_signed_right_shift_arithmetic(self):
        s8 = SInt[8]
        result = s8(-4) >> 1
        assert result.width == 8
        assert result.value == -2


class TestMixedSignedness:
    def test_uint_plus_sint(self):
        result = UInt[13](100) + SInt[16](50)
        assert result.signed == True
        assert result.width == 17
        assert result.value == 150

    def test_uint_plus_sint_negative(self):
        result = UInt[8](10) + SInt[8](-20)
        assert result.signed == True
        assert result.value == -10


class TestBitwiseOps:
    def test_and(self):
        u8 = UInt[8]
        result = u8(0xF0) & u8(0x0F)
        assert result.value == 0
        assert result.width == 8

    def test_or(self):
        u8 = UInt[8]
        result = u8(0xF0) | u8(0x0F)
        assert result.value == 0xFF

    def test_xor(self):
        u8 = UInt[8]
        result = u8(0xFF) ^ u8(0x0F)
        assert result.value == 0xF0

    def test_invert(self):
        u8 = UInt[8]
        result = ~u8(0)
        assert result.value == 0xFF
        assert result.width == 8

    def test_neg(self):
        s8 = SInt[8]
        result = -s8(1)
        assert result.value == -1
        assert result.width == 8
