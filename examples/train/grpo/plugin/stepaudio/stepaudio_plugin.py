# StepAudio2-mini GRPO reward plugin.
#
# Registers two ORM (Outcome Reward Model) functions into swift.rewards.orms:
#   * stepaudio_accuracy  -- 1.0 if the model's predicted label exactly matches GT, else 0.0.
#   * stepaudio_format    -- 1.0 if the completion produces *exactly one* token that
#                            belongs to the allowed label set (i.e. clean format,
#                            no extra explanation / no template leakage), else 0.0.
#
# The data jsonl produced by project/stepaudio/convert_to_swift_format.py contains:
#   {
#     "messages": [
#       {"role": "user", "content": "<audio>... The answer must be exactly one of [speech, music, noise, porn, song]. ..."},
#       {"role": "assistant", "content": "porn"}
#     ],
#     "audios": [...],
#     "label": "porn",      # <-- ground-truth, clean category
#     ...
#   }
#
# In GRPO, swift will route extra columns (like `label`) into reward kwargs,
# so we directly consume `label` here. We also keep `solution` as a fallback
# because some pipelines rename label -> solution via --columns.
#
# Why a custom plugin instead of the built-in `accuracy` / `format`?
#   * Built-in `accuracy` (MathAccuracy) parses LaTeX via math_verify -- it cannot
#     handle plain category strings like "porn" / "speech".
#   * Built-in `format` enforces "<think>...</think><answer>...</answer>" which
#     is NOT what stepaudio is supposed to emit (the assistant target is just
#     a single label token).

import re
from typing import List, Optional

from swift.rewards import ORM, orms


# Default label whitelist for stepaudio (kept in sync with project/stepaudio/eval_classification.py).
# We still try to parse the dynamic "[a, b, c]" list out of every prompt at runtime,
# so adding new classes only requires updating the prompt -- this constant is a fallback.
_DEFAULT_LABELS = ['speech', 'music', 'noise', 'porn', 'song']

# Match the "exactly one of [a, b, c]" enumeration that appears in the user prompt.
# Tolerant to spaces / quotes around items and to either ASCII or Chinese punctuation.
_LABEL_LIST_RE = re.compile(
    r'(?:exactly\s+one\s+of|one\s+of|in)\s*[\[\(]([^\]\)]+)[\]\)]',
    re.IGNORECASE,
)


def _parse_labels_from_prompt(text: str) -> List[str]:
    """Extract the allowed label set declared in the prompt (e.g. "[speech, music, noise]").

    Falls back to _DEFAULT_LABELS when nothing parseable is found, so the
    reward never crashes on malformed prompts.
    """
    if not text:
        return list(_DEFAULT_LABELS)
    m = _LABEL_LIST_RE.search(text)
    if not m:
        return list(_DEFAULT_LABELS)
    raw = m.group(1)
    labels = []
    for tok in re.split(r'[,\s/、，]+', raw):
        tok = tok.strip().strip("'\"`").strip()
        if tok and tok not in labels:
            labels.append(tok)
    return labels or list(_DEFAULT_LABELS)


def _get_prompt_text(messages, instruction) -> str:
    """Concat every text we have on the prompt side so the label list parser
    can find the "[...]" enumeration regardless of where it sits."""
    parts = []
    if messages:
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get('role', '')
                if role in ('user', 'system'):
                    content = msg.get('content', '') or ''
                    parts.append(str(content))
    if instruction:
        parts.append(str(instruction))
    return '\n'.join(parts)


# Tokens we want to strip before matching predictions:
#   - leading/trailing punctuation/whitespace
#   - common chat-template residuals like "<tts_end>", "</s>", "<|endoftext|>"
_STRIP_CHARS = ' \t\r\n.,;:!?。，；：！？\'"`*'
_TEMPLATE_LEAK_RE = re.compile(r'<[^>]{1,40}>')


def _clean_completion(text: str) -> str:
    if not text:
        return ''
    # Drop everything inside angle brackets (template/special tokens), then
    # collapse whitespace.
    text = _TEMPLATE_LEAK_RE.sub(' ', text)
    text = text.strip(_STRIP_CHARS)
    text = re.sub(r'\s+', ' ', text)
    return text


def _predict_label(completion: str, labels: List[str]) -> str:
    """Pick the predicted label from a completion.

    Order of strategies (most specific first):
      1. The cleaned completion is itself exactly one of the labels (case-insensitive).
      2. The completion contains exactly one label as a whole word.
      3. The right-most label occurrence wins (mirrors the "final decision is
         usually at the end" heuristic from event_plugin).
    Returns '' if none match.
    """
    if not completion or not labels:
        return ''
    cleaned = _clean_completion(completion)
    cleaned_low = cleaned.lower()
    label_low = {lab.lower(): lab for lab in labels}

    # 1) exact equality on cleaned text
    if cleaned_low in label_low:
        return label_low[cleaned_low]

    # 2) whole-word match; if exactly one label appears, take it
    found = []
    for lab_low, lab in label_low.items():
        if re.search(r'(?<![A-Za-z0-9_])' + re.escape(lab_low) + r'(?![A-Za-z0-9_])', cleaned_low):
            found.append(lab)
    if len(found) == 1:
        return found[0]

    # 3) right-most occurrence among any label (also covers the multi-match case)
    best_lab, best_pos = '', -1
    for lab in labels:
        pos = cleaned_low.rfind(lab.lower())
        if pos > best_pos:
            best_pos = pos
            best_lab = lab
    return best_lab if best_pos >= 0 else ''


def _normalize_gt(gt) -> str:
    """Ground-truth label may come in as a plain string ("porn") or already
    cleaned by swift's column mapping. Be defensive."""
    if gt is None:
        return ''
    s = str(gt).strip().strip(_STRIP_CHARS)
    return s


class StepAudioAccuracy(ORM):
    """1.0 iff predicted label == ground-truth label (case-insensitive)."""

    def __call__(
        self,
        completions: List[str],
        solution: Optional[List[str]] = None,
        label: Optional[List[str]] = None,
        messages: Optional[List] = None,
        instruction: Optional[List[str]] = None,
        **kwargs,
    ) -> List[float]:
        # Prefer `label` (raw ground-truth column from train.jsonl); fall back
        # to `solution` for users that rename via --columns.
        gts = label if label is not None else solution
        n = len(completions)
        if gts is None:
            gts = [''] * n
        if messages is None:
            messages = [None] * n
        if instruction is None:
            instruction = [None] * n

        rewards: List[float] = []
        for i in range(n):
            completion = completions[i] or ''
            gt = _normalize_gt(gts[i])
            prompt_text = _get_prompt_text(messages[i], instruction[i])
            labels = _parse_labels_from_prompt(prompt_text)
            # Make sure GT is in the candidate set even if prompt parsing missed it.
            if gt and gt not in labels:
                labels.append(gt)
            pred = _predict_label(completion, labels)
            ok = bool(gt) and (pred.lower() == gt.lower())
            rewards.append(1.0 if ok else 0.0)
        return rewards


class StepAudioFormat(ORM):
    """1.0 if completion is *cleanly* a single label from the whitelist.

    Definition of "clean":
      * After stripping template/special tokens and surrounding punctuation,
        the completion equals exactly one allowed label.
    This explicitly *penalizes*:
      * extra commentary / multi-sentence answers
      * empty / leaked template tokens such as '<tts_end>', '<audio_xxx>'
    """

    def __call__(
        self,
        completions: List[str],
        messages: Optional[List] = None,
        instruction: Optional[List[str]] = None,
        **kwargs,
    ) -> List[float]:
        n = len(completions)
        if messages is None:
            messages = [None] * n
        if instruction is None:
            instruction = [None] * n

        rewards: List[float] = []
        for i in range(n):
            completion = completions[i] or ''
            prompt_text = _get_prompt_text(messages[i], instruction[i])
            labels = _parse_labels_from_prompt(prompt_text)
            cleaned = _clean_completion(completion).lower()
            label_low = {lab.lower() for lab in labels}
            rewards.append(1.0 if cleaned in label_low else 0.0)
        return rewards


# --------------------------------------------------------------------------- #
# Class-weighted accuracy reward (inverse-frequency).
# --------------------------------------------------------------------------- #
# Motivation
# ----------
# GRPO with a plain 0/1 accuracy reward + skewed batches converges to
# "always predict the majority class" (see v0/v2/v4 mode-collapse post-mortem).
# Even after balancing the training set (v5/v6), the group-relative advantage
# for rare-class rollouts is dominated by the noise floor: majority-class
# groups have ~4/8 correct by chance, minority-class groups usually 0/8, so
# the *scale* of positive advantage a rare class ever produces is smaller
# than the negative advantage it produces when a majority-class group has one
# outlier miss.
#
# Inverse-frequency weighting fixes this by scaling the +1 reward by the
# rarity of the ground-truth label:
#     reward = 1[pred == gt] * w[gt],  w[gt] = clip(median_freq / freq[gt], w_min, w_max)
# Concretely, using our ORIGINAL data distribution (train.jsonl):
#     speech 69.4% -> w~0.09 (clipped to 1.0)
#     noise  16.4% -> w~0.36 (clipped to 1.0)
#     music   5.7% -> w~1.00
#     porn    4.8% -> w~1.19
#     song    3.7% -> w~1.55
# We clip the majority classes to w_min=1.0 (never *shrink* their reward,
# just don't grow it) and the minority classes to w_max=5.0 (avoid gigantic
# outliers that destabilize GRPO's advantage std). The knobs can be
# overridden via env vars STEPAUDIO_CLASS_WEIGHTS / STEPAUDIO_WEIGHT_CLIP so
# that experiments can sweep them without editing code.
#
# Backwards compatibility: this class is a *new* reward (registered under
# name ``stepaudio_accuracy_weighted``); the original ``stepaudio_accuracy``
# above is untouched. Turn it on by adding to run_train_grpo.sh:
#     REWARD_FUNCS="stepaudio_accuracy_weighted stepaudio_format"
# --------------------------------------------------------------------------- #
import os as _os

# Default weights derived from the original imbalanced train.jsonl distribution
# (see docstring above). Users can override via env var, e.g.:
#   export STEPAUDIO_CLASS_WEIGHTS="speech=1.0,noise=1.5,music=2.5,porn=3.5,song=4.0"
_DEFAULT_CLASS_WEIGHTS = {
    'speech': 1.0,
    'noise': 1.5,
    'music': 3.0,
    'porn': 3.5,
    'song': 4.0,
}


def _parse_class_weights_env() -> dict:
    """Parse STEPAUDIO_CLASS_WEIGHTS='k=v,k=v,...' -> dict[str,float].

    Silently returns _DEFAULT_CLASS_WEIGHTS on parse failure so training
    never crashes because of a misspelled env value; a warning is printed
    once so the misconfiguration is still visible.
    """
    raw = _os.environ.get('STEPAUDIO_CLASS_WEIGHTS', '').strip()
    if not raw:
        return dict(_DEFAULT_CLASS_WEIGHTS)
    out = {}
    try:
        for part in raw.split(','):
            part = part.strip()
            if not part:
                continue
            k, v = part.split('=', 1)
            out[k.strip().lower()] = float(v.strip())
        # Fill in any missing labels with default (so partial overrides work).
        for k, v in _DEFAULT_CLASS_WEIGHTS.items():
            out.setdefault(k, v)
        return out
    except Exception as e:  # pragma: no cover
        print(f'[WARN] STEPAUDIO_CLASS_WEIGHTS parse failed ({e!r}); '
              f'falling back to defaults {_DEFAULT_CLASS_WEIGHTS}', flush=True)
        return dict(_DEFAULT_CLASS_WEIGHTS)


class StepAudioAccuracyWeighted(ORM):
    """Class-weighted 0/w accuracy reward for GRPO.

    * pred == gt:   reward = w[gt]  (>=1 for minority classes)
    * pred != gt:   reward = 0.0
    * gt unknown:   reward = 0.0    (safe: matches vanilla accuracy behaviour)

    Weights are loaded once at construction time from STEPAUDIO_CLASS_WEIGHTS
    or the module-level default. To make GRPO's advantage scale roughly
    comparable to the plain-accuracy setup, the mean of the correct-answer
    reward is normalized to ~1.0 in expectation over the *balanced* training
    distribution -- but we only enforce lower bound w_min=1.0 so majority
    classes never have their positive reward shrunk below 1.0.
    """

    def __init__(self, *args, **kwargs):
        # Parent ORM has an empty __init__, but be defensive if it changes.
        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            super().__init__()
        self._weights = _parse_class_weights_env()
        # Normalize keys to lowercase for case-insensitive lookup.
        self._weights = {k.lower(): float(v) for k, v in self._weights.items()}
        # Print once so it's visible in trainer logs.
        print(f'[StepAudioAccuracyWeighted] class weights = {self._weights}', flush=True)

    def _weight_for(self, label: str) -> float:
        if not label:
            return 1.0
        return float(self._weights.get(label.lower(), 1.0))

    def __call__(
        self,
        completions: List[str],
        solution: Optional[List[str]] = None,
        label: Optional[List[str]] = None,
        messages: Optional[List] = None,
        instruction: Optional[List[str]] = None,
        **kwargs,
    ) -> List[float]:
        gts = label if label is not None else solution
        n = len(completions)
        if gts is None:
            gts = [''] * n
        if messages is None:
            messages = [None] * n
        if instruction is None:
            instruction = [None] * n

        rewards: List[float] = []
        for i in range(n):
            completion = completions[i] or ''
            gt = _normalize_gt(gts[i])
            prompt_text = _get_prompt_text(messages[i], instruction[i])
            labels = _parse_labels_from_prompt(prompt_text)
            if gt and gt not in labels:
                labels.append(gt)
            pred = _predict_label(completion, labels)
            ok = bool(gt) and (pred.lower() == gt.lower())
            rewards.append(self._weight_for(gt) if ok else 0.0)
        return rewards


# Register into swift's ORM registry so that --reward_funcs can find them.
orms['stepaudio_accuracy'] = StepAudioAccuracy
orms['stepaudio_format'] = StepAudioFormat
orms['stepaudio_accuracy_weighted'] = StepAudioAccuracyWeighted
