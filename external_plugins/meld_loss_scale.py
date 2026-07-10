# -*- coding: utf-8 -*-
"""
MELD emotion loss_scale plugin.

Registered strategies:
- ``meld_emotion_b1``:
    Assistant response format used in r1omni_*_self_asr datasets is
        "<transcript>...</transcript>\n<answer>label</answer>"
    We want the model to be supervised **only** on the <answer>...</answer>
    part (label + surrounding tags) while giving zero weight to the
    <transcript>...</transcript> segment and the whitespace between them.

    Rationale:
        v11 (self_asr) matched v10 on training/validation token_acc but
        degraded on test macro-F1 (41.08 vs 53.71 of v10). Root cause hypothesis:
        during training with self-ASR pseudo-labels, the transcript loss
        dominates optimization because it is 5-30x longer than the label,
        making the model over-fit to reproducing the (imperfect) ASR text
        instead of learning emotion. Masking transcript loss removes this
        distraction while keeping the CoT-style format at inference time.

Usage in SFT command:

    --external_plugins /path/to/external_plugins/meld_loss_scale.py \\
    --loss_scale meld_emotion_b1

How it works:
    swift builds the assistant response into a token stream, and asks
    ``loss_scale.get_loss_scale(context, query=...)`` to split it into
    (segments, weights). We use one regex delimiter
        ``<transcript>[\\s\\S]*?</transcript>\\s*``
    with weight 0.0. Every character NOT matched by that regex (i.e. the
    ``<answer>...</answer>`` tail) is left with default weight 1.0.

    See swift.loss_scale.utils.calculate_loss_scale for the split logic;
    it uses split_str_parts_by(regex_mode=True) so the map value is a
    single-element list ``[0.0]`` to mark the matched segment as ignored.

Sanity notes:
    * ``is_binary = True`` so swift's fast path (labels-based CE, liger-kernel
      compatible) is used. All weights are 0 or 1.
    * ``base_strategy='default'`` (only compute loss on assistant responses)
      is inherited from ``LossScale`` — we override ``get_loss_scale`` to
      further mask the transcript region **within** the assistant response.
    * The regex uses ``[\\s\\S]*?`` (non-greedy any-char) instead of ``.*?``
      because ``re.match`` in swift's split runs with ``re.DOTALL`` already,
      but being explicit keeps this plugin portable.
"""

import re
from typing import List, Tuple

from swift.loss_scale.base import LossScale
from swift.loss_scale.mapping import loss_scale_map
from swift.loss_scale.utils import calculate_loss_scale


class MeldEmotionB1LossScale(LossScale):
    """Zero out the transcript CoT segment; keep <answer>...</answer> at weight 1."""

    # All weights we emit are exactly in {0.0, 1.0} -> keep the binary fast path.
    is_binary = True

    # Regex is anchored to the exact tag pair used in
    # data_meld/{train,dev,test}.r1omni_*_self_asr.jsonl assistant contents.
    #   <transcript>Oh my God, ...</transcript>\n<answer>sadness</answer>
    # Matching whitespace suffix ``\s*`` (typically the newline) ensures the
    # newline between the two tags is also zero-weighted, otherwise it stays 1.0
    # and would contribute a tiny but meaningless loss.
    _RESPONSE_LOSS_SCALE_MAP = {
        r'<transcript>[\s\S]*?</transcript>\s*': [0.0],
    }

    def get_loss_scale(self, context, *, query=None, **kwargs) -> Tuple[List[str], List[float]]:
        # ``context`` can be a str (normal path) or a list-of-token-ids (rare,
        # only when previous stages already tokenised). Fall back to the base
        # implementation for the token-id path — masking on token ids would
        # require re-decoding which is not worth the complexity here.
        if isinstance(context, str):
            return calculate_loss_scale(query, context, self._RESPONSE_LOSS_SCALE_MAP)
        return super().get_loss_scale(context, query=query, **kwargs)


# Register strategy names -> class. swift resolves ``--loss_scale meld_emotion_b1``
# via ``swift.loss_scale.mapping.loss_scale_map``. Because
# ``import_external_file`` imports this .py once at startup, the registration
# happens before ``get_loss_scale('meld_emotion_b1')`` is called.
#
# We register two aliases for convenience:
#   * ``meld_emotion_b1``  – matches the design doc terminology (B1 plan).
#   * ``meld_answer_only`` – describes the effect: only <answer>...</answer>
#     participates in the loss.
loss_scale_map['meld_emotion_b1'] = MeldEmotionB1LossScale
loss_scale_map['meld_answer_only'] = MeldEmotionB1LossScale
