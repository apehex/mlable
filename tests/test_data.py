import tensorflow as tf
import tensorflow_datasets as tfds

import mlable.data

# ROPE ########################################################################

class DatasetStatsTest(tf.test.TestCase):
    def setUp(self):
        super(DatasetStatsTest, self).setUp()
        self._dataset = tfds.load('mlqa/en', split='test', as_supervised=False, shuffle_files=True, data_dir='~/.cache/tensorflow/', batch_size=None)
        self._stats_question_all = mlable.data.stats(dataset=self._dataset, features=['question'])
        self._stats_context_question_all = mlable.data.stats(dataset=self._dataset, features=['context', 'question'])
        self._stats_context_question_slice = mlable.data.stats(dataset=self._dataset, features=['context', 'question'], count=128)

    def test_format(self):
        self.assertEqual(set(self._stats_question_all.keys()), {'min', 'avg', 'max'})
        self.assertEqual(set(self._stats_context_question_all.keys()), {'min', 'avg', 'max'})
        self.assertEqual(set(self._stats_context_question_slice.keys()), {'min', 'avg', 'max'})
        assert all(isinstance(__e, int) for __e in self._stats_question_all.values())
        assert all(isinstance(__e, int) for __e in self._stats_context_question_all.values())
        assert all(isinstance(__e, int) for __e in self._stats_context_question_slice.values())

    def test_values(self):
        assert all(__e >= 0 for __e in self._stats_question_all.values())
        assert all(__e >= 0 for __e in self._stats_context_question_all.values())
        assert all(__e >= 0 for __e in self._stats_context_question_slice.values())
        assert self._stats_context_question_all['min'] <= self._stats_question_all['min']
        assert self._stats_context_question_all['avg'] > self._stats_question_all['avg']
        assert self._stats_context_question_all['max'] > self._stats_question_all['max']
        assert self._stats_context_question_all['min'] <= self._stats_context_question_slice['min']
        assert self._stats_context_question_all['max'] >= self._stats_context_question_slice['max']
