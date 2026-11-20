import unittest
from index_guards import classify

ASSERT = ('assert(move _7, "index out of bounds: the length is {} but the index is {}", '
          'const 3_usize, copy _5) -> [success: bb2, unwind continue];')
COMPARE = '_7 = Lt(copy _5, const 3_usize);'


class IndexGuardTests(unittest.TestCase):
    def test_literal_and_remainder_follow_only_adjacent_copies(self):
        self.assertEqual(classify(ASSERT, ['_1 = const 2_usize;', '_5 = copy _1;', COMPARE]),
                         ('literal', 2, 3, 1))
        self.assertEqual(classify(ASSERT, ['_5 = Rem(copy _1, const 2_usize);', COMPARE]),
                         ('remainder', 2, 3, 0))
        self.assertEqual(classify(ASSERT, ['_1 = const 2_usize;', '(*_6) = const 9_usize;',
                                           '_5 = copy _1;', COMPARE])[0],
                         'other fixed-length computation')

    def test_other_registers_conditions_and_named_lengths_are_unclassified(self):
        for condition in [COMPARE.replace('_7', '_4'), COMPARE.replace('_5', '_1'),
                          COMPARE.replace('3_usize', '4_usize'), COMPARE.replace('Lt', 'Le')]:
            self.assertEqual(classify(ASSERT, ['_5 = const 2_usize;', condition])[0],
                             'other fixed-length computation')
        self.assertEqual(classify(ASSERT.replace('move _7', '!move _7'),
                                  ['_5 = const 2_usize;', COMPARE])[0], 'other guard')
        self.assertEqual(classify(ASSERT.replace('const 3_usize', 'copy _9'),
                                  ['_5 = const 2_usize;', COMPARE])[0], 'other guard')

    def test_false_obligations_are_retained_for_the_kernel(self):
        for rhs, expected in [('const 3_usize', ('literal', 3, 3, 0)),
                              ('Rem(copy _1, const 0_usize)', ('remainder', 0, 3, 0)),
                              ('Rem(copy _1, const 4_usize)', ('remainder', 4, 3, 0))]:
            self.assertEqual(classify(ASSERT, [f'_5 = {rhs};', COMPARE]), expected)


if __name__ == '__main__':
    unittest.main()
