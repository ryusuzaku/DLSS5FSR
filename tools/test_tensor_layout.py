"""Regression cases for lost payload bytes, metadata boundaries and FP8 NaNs."""
import math
import struct
import unittest

from classify_dtype import fp8_e4m3
from extract_dlssnr import parse_tensor_table
from resolve_tensors import e4m3, vit_layout


class TensorLayoutTests(unittest.TestCase):
    def test_first_byte_and_last_byte_are_payload(self):
        name = b"block31.layer3.layer"
        payload = bytes.fromhex("fb 2e")
        chunk = len(payload) + 40
        record = (struct.pack("<Q", len(name)) + name + struct.pack("<QQQI", chunk, chunk, len(payload), 1)
                  + payload + struct.pack("<QQI", 0, 1, 1))
        resource = struct.pack("<Q", len(record) + 8) + record
        t, = parse_tensor_table(resource, 0)
        self.assertEqual(resource[t["data_offset"]:t["data_offset"] + 2], payload)
        self.assertEqual(t["data_offset"] + t["data_len"], len(resource))
        self.assertEqual(struct.unpack("<e", payload)[0], 0.10906982421875)

    def test_e4m3_finite_top_exponent(self):
        for decode in (e4m3, fp8_e4m3):
            self.assertEqual(decode(0x78), 256)
            self.assertEqual(decode(0x7e), 448)
            self.assertEqual(decode(0xfe), -448)
            self.assertEqual(sum(math.isnan(decode(i)) for i in range(256)), 2)

    def test_structural_qkv_prefix_beats_plausible_weight_statistics(self):
        payload = struct.pack("<32f", *([1.0] * 32)) + bytes([0x10]) * 3145728
        r = vit_layout("block31.layer2.layer", payload)
        self.assertEqual(r["data_offset"], 128)
        self.assertEqual(r["bytes_of_data"], 3145728)
        with self.assertRaises(ValueError):
            vit_layout("block31.layer2.layer", payload[1:] + b"\0")

    def test_zero_suffix_is_not_skipped_as_a_prefix(self):
        payload = bytes([0x10]) * 4194304 + bytes(16)
        r = vit_layout("block31.layer0.layer", payload)
        self.assertEqual(r["data_offset"], 0)
        self.assertEqual(r["bytes_of_data"], 4194304)
        self.assertIsNone(vit_layout("block0.layer0.layer", payload))


if __name__ == "__main__":
    unittest.main()
