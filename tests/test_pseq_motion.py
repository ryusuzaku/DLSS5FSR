"""Asset-free geometry, native-format and temporal-metric contracts."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from pseq_motion_compare import compare, correspondence, read_native


class MotionGeometry(unittest.TestCase):
    def setUp(self):
        self.h, self.w = 4, 6
        self.old = np.repeat(np.arange(self.w, dtype=np.float32)[None, :, None]/16, self.h, axis=0)
        self.old = np.repeat(self.old, 3, axis=2)
        self.z = np.full((self.h, self.w), 0.2, np.float32)
        self.mv = np.zeros((self.h, self.w, 2), np.float32)
        self.region = np.ones((self.h, self.w), bool)

    def align(self, current, old=None):
        return correspondence(current, self.old, self.mv, self.z, self.z if old is None else old, self.region)

    def test_positive_vector_samples_previous_to_the_right(self):
        self.mv[..., 0] = 1/self.w
        current = np.concatenate((self.old[:, 1:], self.old[:, -1:]), axis=1)
        c = self.align(current)
        self.assertTrue(c['valid'][:, :-1].all())
        self.assertFalse(c['valid'][:, -1].any())
        np.testing.assert_allclose(c['input_error'][c['valid']], 0, atol=1e-6)

    def test_subpixel_translation(self):
        self.mv[..., 0] = 0.25/self.w
        c = self.align(self.old+0.25/16)
        np.testing.assert_allclose(c['input_error'][c['valid']], 0, atol=1e-6)

    def test_current_and_previous_exclusions(self):
        self.mv[..., 0] = 1/self.w
        self.region[:, 3] = False
        c = self.align(self.old)
        self.assertFalse(c['valid'][:, 3].any())
        self.assertFalse(c['valid'][:, 2].any())  # previous sample falls in excluded column
        self.assertTrue(c['valid'][:, 0].all())

    def test_depth_disagreement_and_invalid_depth(self):
        old_z = self.z.copy(); old_z[1, 1] = 0.8; old_z[1, 2] = -1
        c = self.align(self.old, old_z)
        self.assertFalse(c['valid'][1, 1])
        self.assertFalse(c['valid'][1, 2])
        self.assertTrue(c['valid'][2, 2])

    def test_bad_vectors_are_excluded_without_sampler_overflow(self):
        self.mv[0, 0, 0] = np.nan
        self.mv[0, 1, 0] = np.inf
        self.mv[0, 2, 0] = 1e38
        self.mv[0, 3, 0] = 1e30
        with np.errstate(all='raise'):
            c = self.align(self.old)
        self.assertFalse(c['valid'][0, :4].any())
        self.assertTrue(np.isfinite(c['previous']).all())


class NativeAndReplay(unittest.TestCase):
    def test_padded_native_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = np.arange(24, dtype=np.float16).reshape(2, 12)
            path = Path(tmp, 'native.bin')
            path.write_bytes(struct.pack('<3I', 2, 2, 24)+raw.tobytes())
            np.testing.assert_array_equal(read_native(path), raw[:, :8].reshape(2, 2, 4)[..., :3])

    def test_invalid_native_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, 'native.bin')
            for raw in (b'', struct.pack('<3I', 1, 1, 8), struct.pack('<3I', 1, 1, 6)+bytes(6)):
                path.write_bytes(raw)
                with self.assertRaises(ValueError): read_native(path)
            rgba = np.array([np.nan, 0, 0, 1], np.float16)
            path.write_bytes(struct.pack('<3I', 1, 1, 8)+rgba.tobytes())
            with self.assertRaisesRegex(ValueError, 'nonfinite'): read_native(path)

    def replay(self, min_pixels=1, wrong_input=False, tolerances=(2,)):
        h, w = 4, 6
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp, 'capture'); directory.mkdir()
            for n in range(3):
                rgba = np.full((h, w, 4), 0.25, np.float16); rgba[..., 3] = 1
                z = np.full((h, w), 0.2, np.float32)
                mv = np.zeros((h, w, 2), np.float32); mv[..., 0] = 1/w
                raw = struct.pack('<8s7I', b'D5INP001', w, h, 8, w*8, 0, 0, 0)+rgba.tobytes()
                raw += struct.pack('<8s2I', b'D5DEP001', w, h)+z.tobytes()
                raw += struct.pack('<8s2I', b'D5MOV001', w, h)+mv.tobytes()
                (directory/f'{n:03d}.bin').write_bytes(raw)
                for name in ('advected', 'screen_fixed'):
                    out = rgba.copy()
                    out[..., :3] += (np.arange(w)+(n if name == 'advected' else 0))[None, :, None]/64
                    Path(tmp, f'{name}_{n:03d}_native.bin').write_bytes(struct.pack('<3I', w, h, w*8)+out.tobytes())
                    inp = (rgba[..., :3].astype(np.float32)*255+0.5).astype(np.uint8)
                    if wrong_input: inp[0, 0, 0] += 1
                    Path(tmp, f'{name}_{n:03d}_in.rgb').write_bytes(struct.pack('<2I', w, h)+inp.tobytes())
            return compare(directory, {name: Path(tmp, name) for name in ('advected', 'screen_fixed')},
                           count=3, tolerances=tolerances, min_pixels=min_pixels)

    def test_material_aligned_edit_is_stable_but_screen_pattern_crawls(self):
        stats = self.replay()['tolerances']['2']
        moving = stats['moving']
        self.assertEqual(moving['frame_pairs'], 1)
        self.assertEqual(moving['pixels'], 20)
        self.assertAlmostEqual(moving['scores']['advected']['edit_rms'], 0, places=5)
        self.assertAlmostEqual(moving['scores']['screen_fixed']['edit_rms'], 255/64, places=5)
        self.assertEqual(stats['still']['scores'], {})
        self.assertEqual(stats['fast']['scores'], {})

    def test_low_sample_counts_are_empty_not_perfect(self):
        stats = self.replay(min_pixels=100)['tolerances']['2']['moving']
        self.assertEqual(stats['pixels'], 0)
        self.assertIsNone(stats['input_rms'])
        self.assertEqual(stats['scores'], {})

    def test_colour_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'colour differs'):
            self.replay(wrong_input=True)

    def test_close_tolerances_do_not_merge_sample_counts(self):
        stats = self.replay(tolerances=(2, 2.00000001))['tolerances']
        self.assertEqual(len(stats), 2)
        self.assertTrue(all(group['moving']['pixels'] == 20 for group in stats.values()))


if __name__ == '__main__':
    unittest.main()
