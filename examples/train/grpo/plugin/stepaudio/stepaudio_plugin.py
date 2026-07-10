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

# --------------------------------------------------------------------------- #
# Recall-aware reward: penalize "lazy majority prediction".
# --------------------------------------------------------------------------- #
# Motivation
# ----------
# GRPO v0 on MELD collapsed to over-predicting `neutral` (majority): even after
# switching to weighted accuracy, the *only* way for GRPO to lose reward on a
# rare-class rollout is "not perfectly matching gt", so the policy learned the
# safe strategy "always guess neutral -- majority classes will still hit, and
# the rare-class group std tends to be small enough that gradients don't fight
# back". The training-time neutral over-prediction rose from 70.8% (SFT) to
# 82.3% (GRPO-200), and disgust recall collapsed from 45.6% to 11.8%.
#
# The plain fix is asymmetric: keep the positive reward path exactly as
# `stepaudio_accuracy_weighted`, but add a *negative* reward when the policy
# tries to hedge with `neutral` (or any configured "lazy" label) while the gt
# is a rare class. Concretely:
#
#   reward =  w[gt]                            if pred == gt   (positive path)
#         =  -lazy_penalty                     if pred in LAZY and gt not in LAZY
#         =  0.0                               otherwise
#
# Design choices:
#   * We do NOT penalize the LAZY class when it *is* the ground-truth --- that
#     would break majority learning entirely. This asymmetry is the whole
#     point: we only punish "guessing neutral when the truth is not neutral".
#   * We do NOT penalize other wrong predictions (e.g. gt=anger, pred=joy) with
#     -lazy_penalty --- those are honest mistakes, they already earn 0.0 and
#     shouldn't be pulled further below by this term. Only the hedging
#     behaviour is targeted.
#   * The magnitude of the penalty is configurable via
#         STEPAUDIO_LAZY_PENALTY (default 0.5)
#     and the set of "lazy" labels via
#         STEPAUDIO_LAZY_LABELS   (default "neutral")
#     Both can be sweeped without editing code.
#
# Backwards compat: this is a NEW reward name (`stepaudio_accuracy_recall`);
# existing three rewards are untouched. Turn it on by adding to your shell:
#     REWARD_FUNCS="stepaudio_accuracy_recall"
# --------------------------------------------------------------------------- #

def _parse_lazy_labels_env() -> set:
    """Parse STEPAUDIO_LAZY_LABELS='neutral,other' -> set of lowercase labels.

    Empty / missing env var falls back to {'neutral'}. Silent on parse error
    (fall back to default) but prints a warning to trainer logs.
    """
    raw = _os.environ.get('STEPAUDIO_LAZY_LABELS', '').strip()
    if not raw:
        return {'neutral'}
    try:
        out = set()
        for tok in raw.split(','):
            tok = tok.strip().lower()
            if tok:
                out.add(tok)
        return out or {'neutral'}
    except Exception as e:  # pragma: no cover
        print(f'[WARN] STEPAUDIO_LAZY_LABELS parse failed ({e!r}); '
              f'falling back to {{neutral}}', flush=True)
        return {'neutral'}

def _parse_lazy_penalty_env() -> float:
    """Parse STEPAUDIO_LAZY_PENALTY float. Default 0.5.

    Values <=0 effectively disable the penalty (equivalent to
    stepaudio_accuracy_weighted).
    """
    raw = _os.environ.get('STEPAUDIO_LAZY_PENALTY', '').strip()
    if not raw:
        return 0.5
    try:
        v = float(raw)
        if v < 0:
            print(f'[WARN] STEPAUDIO_LAZY_PENALTY={v} < 0; taking abs value.', flush=True)
            v = abs(v)
        return v
    except Exception as e:  # pragma: no cover
        print(f'[WARN] STEPAUDIO_LAZY_PENALTY parse failed ({e!r}); using 0.5', flush=True)
        return 0.5

def _parse_lazy_strict_env() -> bool:
    """Parse STEPAUDIO_LAZY_STRICT bool. Default False.

    When STRICT=False (default, backwards-compatible):
        pred in LAZY and gt NOT in LAZY   -> -penalty
        pred in LAZY and gt IN LAZY (mismatch, e.g. gt=neutral,pred=joy) -> 0.0
    When STRICT=True (v4+):
        pred in LAZY and pred != gt       -> -penalty  (regardless of gt in LAZY)
    Rationale: v3 on MELD showed 'joy explosion' (recall 34.6 -> 48.5) because
    with LAZY={neutral,joy} the non-strict rule exempts gt=neutral,pred=joy from
    penalty (gt is in LAZY so we skip). joy therefore steals mass from neutral
    at zero cost. STRICT mode closes this loophole: any LAZY prediction that
    doesn't match gt is penalized, forcing policy to only pick LAZY labels when
    the GT actually is that LAZY label.
    """
    raw = _os.environ.get('STEPAUDIO_LAZY_STRICT', '').strip().lower()
    if not raw:
        return False
    return raw in ('1', 'true', 'yes', 'on')

class StepAudioAccuracyRecall(ORM):
    """Asymmetric class-weighted reward that also punishes lazy majority guesses.

    Non-strict mode (STEPAUDIO_LAZY_STRICT=0, default; backwards compatible):
      * pred == gt:                                reward = w[gt]      (positive)
      * pred in LAZY_LABELS and gt not in LAZY:    reward = -penalty
      * any other mismatch:                        reward = 0.0
      * gt unknown / empty:                        reward = 0.0

    Strict mode (STEPAUDIO_LAZY_STRICT=1, v4+; closes symmetric-exemption bug):
      * pred == gt:                                reward = w[gt]      (positive)
      * pred in LAZY_LABELS and pred != gt:        reward = -penalty   (NEW)
      * any other mismatch:                        reward = 0.0
      * gt unknown / empty:                        reward = 0.0
    Note: in strict mode, gt=joy,pred=neutral is penalized (since neutral is
    LAZY and pred != gt). This is intentional: it suppresses the reverse
    hedging direction too. But pred=neutral when gt=neutral remains
    correct-path (+w[gt]), so majority learning is preserved.

    Env vars (see module docstring above):
        STEPAUDIO_CLASS_WEIGHTS    (same schema as StepAudioAccuracyWeighted)
        STEPAUDIO_LAZY_LABELS      comma-separated, default "neutral"
        STEPAUDIO_LAZY_PENALTY     float, default 0.5
        STEPAUDIO_LAZY_STRICT      bool (0/1), default 0

    Compatibility notes:
      * Rewards can be negative here, which is fine for GRPO advantage
        computation --- the trainer computes (r - mean)/std within each group,
        so a negative baseline just shifts the origin, it doesn't break
        anything. The only place negative rewards care is
        `scale_rewards=none`: they DO flow directly into the loss then, which
        is exactly what we want (majority-hedging directly hurts).
      * With `scale_rewards=group`, the -penalty term is partially normalized
        away when all 8 rollouts in a group are LAZY guesses (std=0 for the
        penalty component). If you rely on this reward, prefer
        `scale_rewards=none` or `scale_rewards=batch`.
    """

    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            super().__init__()
        self._weights = {k.lower(): float(v) for k, v in _parse_class_weights_env().items()}
        self._lazy_labels = _parse_lazy_labels_env()
        self._lazy_penalty = _parse_lazy_penalty_env()
        self._lazy_strict = _parse_lazy_strict_env()
        print(
            f'[StepAudioAccuracyRecall] class_weights={self._weights} '
            f'lazy_labels={sorted(self._lazy_labels)} lazy_penalty={self._lazy_penalty} '
            f'lazy_strict={self._lazy_strict}',
            flush=True,
        )

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
            gt_low = gt.lower() if gt else ''
            pred_low = pred.lower() if pred else ''

            if gt_low and pred_low == gt_low:
                # correct -> positive class-weighted reward
                rewards.append(self._weight_for(gt))
            elif pred_low in self._lazy_labels and gt_low and pred_low != gt_low and (
                    self._lazy_strict or gt_low not in self._lazy_labels):
                # v4: STRICT mode drops the "gt in LAZY" exemption to close the
                # symmetric-exemption bug (e.g. gt=neutral,pred=joy now penalized
                # even though gt is also in LAZY). Non-strict keeps the original
                # "gt not in LAZY" filter for backwards compatibility.
                rewards.append(-self._lazy_penalty)
            else:
                # honest miss / unknown gt -> zero (matches baseline behaviour)
                rewards.append(0.0)
        return rewards

# Register into swift's ORM registry so that --reward_funcs can find them.
orms['stepaudio_accuracy'] = StepAudioAccuracy
orms['stepaudio_format'] = StepAudioFormat
orms['stepaudio_accuracy_weighted'] = StepAudioAccuracyWeighted
orms['stepaudio_accuracy_recall'] = StepAudioAccuracyRecall

# --------------------------------------------------------------------------- #
# Transcript-WER reward: reward the model for accurate self-transcription.
# --------------------------------------------------------------------------- #
# Motivation
# ----------
# The r1omni_self_asr data format asks the model to first transcribe the
# audio inside <transcript>...</transcript> and only then emit the emotion
# label inside <answer>...</answer>. Empirically (MELD test on v10/v11 SFT),
# giving the model *external* ASR text lifts accuracy by ~10pp (from 59% to
# 69%) -- meaning the ASR content itself carries strong emotion signal. We
# therefore want the GRPO policy to be *rewarded* for producing a faithful
# transcript, not just the correct label. This closes the loop of
# "self-ASR -> use its own transcript -> better label decision".
#
# Reward shape (deliberately CONTINUOUS instead of a hard-threshold bonus):
#     wer = Levenshtein(pred_tokens, gold_tokens) / max(len(gold_tokens), 1)
#     r   = clip(1.0 - wer, 0.0, 1.0)
# Rationale for continuous over threshold:
#   * Threshold ("bonus if wer < T") creates a discontinuity at wer=T --> the
#     policy oscillates at the boundary; wer=0 and wer=T-eps get the same
#     reward, so once the model just barely crosses T there is no gradient
#     pulling it further; below T there is no gradient pulling it up either.
#   * `1 - wer` is monotone, bounded in [0, 1], differentiable-friendly for
#     advantage estimation, and directly interpretable.
#
# Gating:
#   * If the completion doesn't contain a <transcript>...</transcript> block,
#     reward = 0.0  (do NOT return negative; we don't want to over-punish
#     "went straight to the label" -- the accuracy reward already handles
#     the label side, WER is strictly a positive-only bonus).
#   * If `asr_text` (gold transcript) is missing / empty in the jsonl row,
#     reward = 0.0  (safe fallback for legacy datasets without ASR).
#
# Data plumbing:
#   swift automatically routes extra jsonl columns into reward kwargs, so as
#   long as the training jsonl (produced by
#   project/stepaudio/optimize_user_prompt_r1omni_self_asr.py) contains a
#   top-level `asr_text` field, we receive it as a list[str] here.
#
# Env knobs (all optional):
#   STEPAUDIO_WER_TOKENIZATION  'word' (default) | 'char'  -- unit of edit
#                                distance. Word-level is more forgiving to
#                                spelling / punctuation which is irrelevant
#                                for the downstream emotion task.
#   STEPAUDIO_WER_LOWERCASE     '1' (default) | '0'  -- lowercase both sides
#                                before comparison.
#   STEPAUDIO_WER_STRIP_PUNCT   '1' (default) | '0'  -- strip common
#                                punctuation before tokenization (word mode).
#
# Backwards compat: NEW reward name, additive registration; no existing
# reward is modified. Enable via run_train_grpo_meld.sh:
#     REWARD_FUNCS="stepaudio_accuracy_recall stepaudio_transcript_wer"
#     REWARD_WEIGHTS="1.0 0.3"
# --------------------------------------------------------------------------- #

# Extract the *first* <transcript>...</transcript> block; tolerant to
# missing closing tag by falling back to "<transcript> ... <answer>" span.
_TRANSCRIPT_RE = re.compile(
    r'<\s*transcript\s*>(.*?)<\s*/\s*transcript\s*>',
    re.IGNORECASE | re.DOTALL,
)
_TRANSCRIPT_OPEN_ONLY_RE = re.compile(
    r'<\s*transcript\s*>(.*?)(?:<\s*answer\s*>|$)',
    re.IGNORECASE | re.DOTALL,
)
_WER_PUNCT_RE = re.compile(r"[.,!?;:\"'`()\[\]{}<>*_~/\\|@#$%^&+=]+")


def _extract_transcript(text: str) -> str:
    """Pull the text inside <transcript>...</transcript>. Returns '' if not present.

    Falls back to "<transcript>...<answer>" span (no closing tag but the
    label section still starts) to be resilient against MAX_COMPLETION_LENGTH
    truncation right after the transcript body.
    """
    if not text:
        return ''
    m = _TRANSCRIPT_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _TRANSCRIPT_OPEN_ONLY_RE.search(text)
    if m:
        return m.group(1).strip()
    return ''


def _wer_tokenize(text: str, tokenization: str, lowercase: bool, strip_punct: bool):
    """Return list of tokens for edit-distance computation."""
    if text is None:
        return []
    s = str(text)
    if lowercase:
        s = s.lower()
    if tokenization == 'char':
        # keep original chars (still may strip whitespace).
        if strip_punct:
            s = _WER_PUNCT_RE.sub('', s)
        return [c for c in s if not c.isspace()]
    # word-level (default)
    if strip_punct:
        s = _WER_PUNCT_RE.sub(' ', s)
    return [tok for tok in re.split(r'\s+', s.strip()) if tok]


def _levenshtein(a, b) -> int:
    """Standard O(len(a) * len(b)) Levenshtein on token sequences.

    For MELD-style utterances (~30-60 words) this is ~a few thousand ops per
    pair, entirely negligible relative to GRPO's per-step generation cost.
    """
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    # Two-row rolling buffer.
    prev = list(range(lb + 1))
    cur = [0] * (lb + 1)
    for i in range(1, la + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + cost, # substitution
            )
        prev, cur = cur, prev
    return prev[lb]


def _wer(pred: str, gold: str, tokenization: str, lowercase: bool, strip_punct: bool) -> float:
    """Word (or char) error rate: edits / max(1, len(gold_tokens)).

    Returns 1.0 (worst) if gold is empty AND pred is non-empty (all insertion),
    0.0 if both empty.
    """
    p_tok = _wer_tokenize(pred, tokenization, lowercase, strip_punct)
    g_tok = _wer_tokenize(gold, tokenization, lowercase, strip_punct)
    if not g_tok:
        return 0.0 if not p_tok else 1.0
    d = _levenshtein(p_tok, g_tok)
    return d / float(len(g_tok))


class StepAudioTranscriptWER(ORM):
    """Continuous WER-based bonus reward for the <transcript>...</transcript> segment.

    reward = clip(1.0 - WER(pred_transcript, gold_asr_text), 0.0, 1.0)
    reward = 0.0 if either <transcript> tag is missing in completion or
             `asr_text` / `transcript` field is missing in the training row.

    Env knobs (STEPAUDIO_WER_TOKENIZATION / _LOWERCASE / _STRIP_PUNCT). See
    module-level docstring above.
    """

    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            super().__init__()
        self._tokenization = _os.environ.get('STEPAUDIO_WER_TOKENIZATION', 'word').strip().lower()
        if self._tokenization not in ('word', 'char'):
            print(f'[WARN] STEPAUDIO_WER_TOKENIZATION={self._tokenization!r} invalid; '
                  f'falling back to "word".', flush=True)
            self._tokenization = 'word'
        self._lowercase = _os.environ.get('STEPAUDIO_WER_LOWERCASE', '1').strip().lower() not in ('0', 'false', 'no', 'off', '')
        self._strip_punct = _os.environ.get('STEPAUDIO_WER_STRIP_PUNCT', '1').strip().lower() not in ('0', 'false', 'no', 'off', '')
        print(
            f'[StepAudioTranscriptWER] tokenization={self._tokenization} '
            f'lowercase={self._lowercase} strip_punct={self._strip_punct}',
            flush=True,
        )

    def __call__(
        self,
        completions: List[str],
        asr_text: Optional[List[str]] = None,
        transcript: Optional[List[str]] = None,
        solution: Optional[List[str]] = None,
        label: Optional[List[str]] = None,
        messages: Optional[List] = None,
        instruction: Optional[List[str]] = None,
        **kwargs,
    ) -> List[float]:
        # Prefer `asr_text` (native column name from
        # optimize_user_prompt_r1omni_self_asr.py); fall back to `transcript`
        # for datasets renamed via --columns; further fall back to `solution`
        # for math-style pipelines. `label` is ignored here (this reward
        # doesn't grade the emotion label -- accuracy_recall does).
        golds = asr_text if asr_text is not None else transcript
        if golds is None:
            golds = solution
        n = len(completions)
        if golds is None:
            golds = [''] * n

        rewards: List[float] = []
        for i in range(n):
            comp = completions[i] or ''
            gold = str(golds[i]) if golds[i] is not None else ''
            pred = _extract_transcript(comp)
            if not pred:
                rewards.append(0.0)
                continue
            if not gold.strip():
                rewards.append(0.0)
                continue
            wer = _wer(pred, gold, self._tokenization, self._lowercase, self._strip_punct)
            r = 1.0 - wer
            if r < 0.0:
                r = 0.0
            elif r > 1.0:
                r = 1.0
            rewards.append(float(r))
        return rewards


orms['stepaudio_transcript_wer'] = StepAudioTranscriptWER
