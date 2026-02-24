from rtlmod import UInt


class TestUIntConstruction:
    def test_type_creation(self):
        u8 = UInt[8]
        assert u8.width == 8
        assert u8.signed == False

    def test_scalar_creation(self):
        u8 = UInt[8]
        x = u8(42)
        assert x.value == 42
        assert x.width == 8
        assert x.signed == False

    def test_masking_on_construction(self):
        u8 = UInt[8]
        x = u8(256)
        assert x.value == 0

    def test_masking_large_value(self):
        u8 = UInt[8]
        x = u8(0x1FF)
        assert x.value == 0xFF

    def test_negative_wraps(self):
        u8 = UInt[8]
        x = u8(-1)
        assert x.value == 255

    def test_type_caching(self):
        assert UInt[8] is UInt[8]

    def test_different_widths_different_types(self):
        assert UInt[8] is not UInt[16]

    def test_equality(self):
        u8 = UInt[8]
        assert u8(42) == u8(42)
        assert u8(42) != u8(43)

    def test_equality_cross_width(self):
        assert UInt[8](42) != UInt[16](42)

    def test_zero(self):
        u8 = UInt[8]
        x = u8(0)
        assert x.value == 0

    def test_max_value(self):
        u8 = UInt[8]
        x = u8(255)
        assert x.value == 255

    def test_wide_type(self):
        u258 = UInt[258]
        x = u258((1 << 258) - 1)
        assert x.value == (1 << 258) - 1

    def test_wide_masking(self):
        u258 = UInt[258]
        x = u258(1 << 258)
        assert x.value == 0
