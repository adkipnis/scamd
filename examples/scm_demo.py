# """Generate SCM samples with randomized configuration draws.
#
# This script repeatedly samples SCM and Posthoc pipelines to sanity check stability.
# """
#
# import sys
# from pathlib import Path
#
# from tqdm import tqdm
#
# ROOT = Path(__file__).resolve().parents[1]
# if str(ROOT) not in sys.path:
#     sys.path.insert(0, str(ROOT))
#
# from scamd.pool import getActivations
# from scamd.scm import Posthoc, SCM
# from scamd.utils import getRng, logUniform, setSeed
#
#
# if __name__ == '__main__':
#     setSeed(0)
#     batches = 32
#     activations = getActivations()
#
#     for _ in tqdm(range(batches)):
#         config = {
#             'n_samples': 512,
#             'n_features': 5,
#             'n_causes': logUniform(getRng(), 2, 12, round=True),
#             'cause_dist': getRng().choice(['uniform', 'normal', 'mixed']),
#             'fixed': getRng().choice([True, False]),
#             'n_hidden': logUniform(getRng(), 5, 30, round=True, add=4),
#             'n_layers': getRng().integers(8, 32),
#             'activation': getRng().choice(activations),
#             'contiguous': getRng().choice([True, False]),
#             'blockwise': getRng().choice([True, False]),
#         }
#         scm = SCM(**config)
#         ph = Posthoc(**config)
#         x = scm.sample()
#         _ = ph(x)
