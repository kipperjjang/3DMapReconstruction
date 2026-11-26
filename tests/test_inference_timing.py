import unittest
from unittest.mock import patch

from neural_scene.inference_timing import InferenceTiming


class InferenceTimingTests(unittest.TestCase):
    def test_cuda_work_is_completed_before_reading_clock(self):
        events = []
        ticks = iter([1., 1.025, 2., 2.075])

        def clock():
            events.append('clock')
            return next(ticks)

        timing = InferenceTiming('cuda:1')
        with patch('torch.cuda.synchronize', side_effect=lambda device: events.append(device)), \
                patch('neural_scene.inference_timing.time.perf_counter', side_effect=clock):
            for _ in range(2):
                with timing.measure():
                    events.append('inference')
        self.assertEqual(events, ['cuda:1', 'clock', 'inference', 'cuda:1', 'clock'] * 2)
        self.assertEqual(timing.frames, 2)
        self.assertAlmostEqual(timing.total_s, .1)
        self.assertAlmostEqual(timing.summary()['mean_inference_ms'], 50.)

    def test_empty_and_failed_measurements_do_not_count_frames(self):
        timing = InferenceTiming('cpu')
        self.assertIsNone(timing.summary()['mean_inference_ms'])
        with self.assertRaises(RuntimeError):
            with timing.measure():
                raise RuntimeError('failed inference')
        self.assertEqual(timing.frames, 0)
        self.assertEqual(timing.total_s, 0)
