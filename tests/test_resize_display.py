from rtlmod import UInt, SInt


class TestResize:
    def test_truncate_unsigned(self):
        u16 = UInt[16]
        result = u16(0xABCD).resize(8)
        assert result.width == 8
        assert result.value == 0xCD
        assert result.signed == False

    def test_truncate_signed(self):
        s16 = SInt[16]
        result = s16(0x7FFF).resize(8)
        assert result.width == 8
        assert result.signed == True

    def test_widen_unsigned(self):
        u8 = UInt[8]
        result = u8(0xAB).resize(16)
        assert result.width == 16
        assert result.value == 0xAB

    def test_widen_signed_positive(self):
        s8 = SInt[8]
        result = s8(42).resize(16)
        assert result.width == 16
        assert result.value == 42

    def test_widen_signed_negative(self):
        s8 = SInt[8]
        result = s8(-1).resize(16)
        assert result.width == 16
        assert result.value == -1

    def test_saturate_unsigned(self):
        u16 = UInt[16]
        result = u16(300).resize(8, round='saturate')
        assert result.value == 255

    def test_saturate_signed_positive(self):
        s16 = SInt[16]
        result = s16(200).resize(8, round='saturate')
        assert result.value == 127

    def test_saturate_signed_negative(self):
        s16 = SInt[16]
        result = s16(-200).resize(8, round='saturate')
        assert result.value == -128


class TestSignExtend:
    def test_sign_extend_positive(self):
        u8 = UInt[8]
        result = u8(42).sign_extend(16)
        assert result.width == 16
        assert result.signed == True
        assert result.value == 42

    def test_sign_extend_with_msb_set(self):
        u8 = UInt[8]
        result = u8(0x80).sign_extend(16)
        assert result.value == -128

    def test_sign_extend_signed(self):
        s8 = SInt[8]
        result = s8(-5).sign_extend(16)
        assert result.value == -5
        assert result.width == 16


class TestDisplay:
    def test_uint_str(self):
        u8 = UInt[8]
        assert str(u8(42)) == "u8'd42"

    def test_sint_str(self):
        s16 = SInt[16]
        assert str(s16(-1)) == "s16'd-1"

    def test_uint_hex(self):
        assert UInt[8](0xAB).hex == "8'hab"

    def test_sint_hex(self):
        assert SInt[16](-1).hex == "16'shffff"

    def test_uint_bin(self):
        u4 = UInt[4]
        assert u4(0b1010).bin == "4'b1010"

    def test_sint_bin(self):
        s4 = SInt[4]
        assert s4(-1).bin == "4'sb1111"

    def test_repr(self):
        assert repr(UInt[8](42)) == "UInt[8](42)"
        assert repr(SInt[16](-1)) == "SInt[16](-1)"

    def test_wide_hex(self):
        u258 = UInt[258]
        x = u258(0xFF)
        h = x.hex
        assert h.startswith("258'h")
        assert h.endswith("ff")
