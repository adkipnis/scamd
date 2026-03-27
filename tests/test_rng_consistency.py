import unittest

from scamd.utils import setSeed


def _contains_forbidden_rng_calls(text: str) -> bool:
    forbidden = (
        'np.random.',
        'random.',
    )
    return any(token in text for token in forbidden)


class TestRngConsistency(unittest.TestCase):
    def test_no_np_random_or_python_random_in_package(self) -> None:
        setSeed(0)
        paths = [
            'scamd/basic.py',
            'scamd/causes.py',
            'scamd/gp.py',
            'scamd/posthoc.py',
            'scamd/scm.py',
            'scamd/pool.py',
            'scamd/meta.py',
        ]
        offenders: list[str] = []
        for path in paths:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            if _contains_forbidden_rng_calls(text):
                offenders.append(path)
        self.assertEqual(offenders, [])


if __name__ == '__main__':
    unittest.main()
