from pathlib import Path
import tempfile
import unittest

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from neural_scene.training import log_scalars, tensorboard_writer


class TensorBoardTests(unittest.TestCase):
    def test_events_are_readable_after_exception_and_missing_distances_are_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                with tensorboard_writer(directory) as writer:
                    log_scalars(writer, 'validation', {'model': {
                        'f1': 0.5, 'distance': None, 'invalid': float('nan')}}, 3)
                    raise RuntimeError('interrupted')
            events = EventAccumulator(directory).Reload()
            self.assertEqual(events.Tags()['scalars'], ['validation/model/f1'])
            self.assertEqual(events.Scalars('validation/model/f1')[0].step, 3)

    def test_disabled_logging_creates_no_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'tensorboard'
            with tensorboard_writer(output, enabled=False) as writer:
                log_scalars(writer, 'train', {'loss': 1.0}, 1)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
