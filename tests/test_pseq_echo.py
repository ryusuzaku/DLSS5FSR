"""Analytical contracts for the optional input-transient diagnostic.

Run: python -m unittest discover -s tests -p test_pseq_echo.py
Requires NumPy, as does the replay comparison tool.
"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import tempfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from pseq_trail_compare import compare, echo_terms, echo_summary


class EchoContracts(unittest.TestCase):
    def setUp(self):
        self.g = np.array([[3., 0., 0.], [-3., 0., 0.]])

    def summarize(self, z, scale=1):
        return echo_summary(echo_terms(self.g, z), self.g.size, scale)

    def test_identity_preserves_input(self):
        s = self.summarize(self.g)
        self.assertAlmostEqual(s['coherent_input_gain'], 1)
        self.assertAlmostEqual(s['output_rms'], s['input_rms'])
        self.assertEqual(s['edit_transient_rms'], 0)

    def test_amplification_and_cancellation(self):
        self.assertAlmostEqual(self.summarize(2*self.g)['coherent_input_gain'], 2)
        s = self.summarize(np.zeros_like(self.g))
        self.assertAlmostEqual(s['coherent_input_gain'], 0)
        self.assertAlmostEqual(s['output_rms'], 0)
        reversed_component = self.summarize(-2*self.g)
        self.assertAlmostEqual(reversed_component['coherent_input_gain'], -2)
        self.assertAlmostEqual(reversed_component['output_rms'], 2*reversed_component['input_rms'])

    def test_orthogonal_change_is_not_coherent_amplification(self):
        independent = np.array([[0., 3., 0.], [0., -3., 0.]])
        s = self.summarize(self.g+independent)
        self.assertAlmostEqual(s['coherent_input_gain'], 1)
        self.assertAlmostEqual(s['orthogonal_edit_rms'], s['input_rms'])
        self.assertAlmostEqual(s['output_rms'], np.sqrt(2)*s['input_rms'])

    def test_strength_match_removes_pure_scaling(self):
        weak = self.summarize(1.5*self.g, 2)
        strong = self.summarize(2*self.g)
        self.assertAlmostEqual(weak['matched_coherent_input_gain'], strong['coherent_input_gain'])
        self.assertAlmostEqual(weak['matched_output_rms'], strong['output_rms'])

    def test_pooling_is_weighted_by_input_energy(self):
        terms = echo_terms(self.g, 2*self.g)+echo_terms(2*self.g, 6*self.g)
        s = echo_summary(terms, self.g.size*2)
        self.assertAlmostEqual(s['coherent_input_gain'], (2+4*3)/5)

    def test_weak_or_absent_input_has_no_direction(self):
        s = echo_summary(echo_terms(np.zeros_like(self.g), self.g), self.g.size)
        self.assertIsNone(s['coherent_input_gain'])
        self.assertIsNone(s['matched_coherent_input_gain'])
        self.assertGreater(s['output_rms'], 0)
        weak = echo_summary(echo_terms(0.01*self.g, self.g), self.g.size)
        self.assertIsNone(weak['coherent_input_gain'])
        self.assertEqual(echo_summary([0, 0, 0], 0), {})
        self.assertIsNone(self.summarize(self.g, None)['matched_output_rms'])

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'shapes differ'):
            echo_terms(self.g, self.g[:1])


class EchoReplayContracts(unittest.TestCase):
    def replay(self, echo=True, colour_mismatch=False, style_offset=False, count=18):
        # One 512-pixel uncover event, still camera and a settled input tail.
        # Input transient=20/255. Weak/strong edits add 10/20 respectively.
        def frame(path):
            n = int(Path(path).stem)
            pic = np.full((64, 64, 3), 64/255, np.float32)
            if n == 1:
                pic[:16, :32] = 84/255
            mv = np.zeros((64, 64, 2), np.float32)
            if n == 0:
                mv[:16, :32, 0] = 8/64
            return pic, None, mv

        def replay_rgb(path):
            name, number, kind = Path(path).stem.rsplit('_', 2)
            n = int(number)
            rgb = (frame(f'{n}.bin')[0]*255+0.5).astype(np.uint8)
            if kind == 'out':
                if style_offset:
                    rgb += 5 if name == 'weak' else 7
                if n == 1:
                    rgb[:16, :32] += 10 if name == 'weak' else 20
            elif colour_mismatch:
                rgb[0, 0, 0] += 1
            return rgb

        with tempfile.TemporaryDirectory() as folder:
            for n in range(count):
                Path(folder, f'{n:03d}.bin').touch()
            with patch('pseq_trail_compare.read_frame', side_effect=frame), patch('pseq_trail_compare.read_rgb8', side_effect=replay_rgb):
                return compare(folder, {'weak': 'weak', 'strong': 'strong'}, 'strong', count=count, echo=echo)

    def test_full_replay_projection_and_strength_control(self):
        report = self.replay()
        self.assertEqual(report['input_echo']['frame_pairs'], 1)
        self.assertEqual(report['input_echo']['uncovered_pixels'], 512)
        weak = report['input_echo']['scores']['weak']['0']
        strong = report['input_echo']['scores']['strong']['0']
        self.assertAlmostEqual(weak['coherent_input_gain'], 1.5, places=5)
        self.assertAlmostEqual(strong['coherent_input_gain'], 2, places=5)
        self.assertAlmostEqual(weak['edit_match_scale'], 2)
        self.assertAlmostEqual(weak['matched_coherent_input_gain'], 2, places=5)

    def test_each_replay_uses_its_own_settled_output(self):
        scores = self.replay(style_offset=True)['input_echo']['scores']
        self.assertAlmostEqual(scores['weak']['0']['coherent_input_gain'], 1.5, places=5)
        self.assertAlmostEqual(scores['strong']['0']['coherent_input_gain'], 2, places=5)

    def test_optional_mode_preserves_existing_metrics(self):
        enabled = self.replay()
        enabled.pop('input_echo')
        self.assertEqual(enabled, self.replay(echo=False))

    def test_incompatible_input_colour_space_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'capture colour space'):
            self.replay(colour_mismatch=True)

    def test_no_eligible_samples_is_not_a_zero_gain(self):
        self.assertEqual(self.replay(count=10)['input_echo']['scores'], {})


if __name__ == '__main__':
    unittest.main()
