import re
from typing import List, Optional

from swift.rewards import ORM, orms


# Regex used to parse the "# 类别列表" block from instruction / system prompt.
# Matches lines like "- 色情" / "- 社会价值观贬损（动态新增）".
_CATEGORY_LIST_RE = re.compile(
    r'#\s*类别列表\s*\n((?:\s*[\-\*]\s*[^\n]+\n?)+)',
    re.MULTILINE,
)
_CATEGORY_LINE_RE = re.compile(r'^\s*[\-\*]\s*([^\n（(]+?)\s*(?:[（(][^）)]*[）)])?\s*$', re.MULTILINE)


def _extract_categories_from_text(text: str) -> List[str]:
    """Parse the dynamic category list embedded in the prompt's instruction.

    Returns a list of category names (strings, may include Chinese characters).
    """
    if not text:
        return []
    m = _CATEGORY_LIST_RE.search(text)
    block = m.group(1) if m else text  # fall back to whole text
    cats = []
    for line in block.splitlines():
        m2 = _CATEGORY_LINE_RE.match(line)
        if m2:
            cat = m2.group(1).strip()
            if cat and cat not in cats:
                cats.append(cat)
    return cats


def _get_prompt_text(messages, instruction) -> str:
    """Concatenate user-side text from messages / instruction so that we can
    parse the dynamic category list out of it."""
    parts = []
    if messages:
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get('role', '')
                content = msg.get('content', '') or ''
                if role in ('user', 'system'):
                    parts.append(str(content))
    if instruction:
        parts.append(str(instruction))
    return '\n'.join(parts)


def _find_category_in_completion(completion: str, categories: List[str]) -> str:
    """Pick the model's predicted category from the completion.

    Strategy:
      1. If the completion explicitly says "类别：XXX" / "分类：XXX" / "**类别**：XXX",
         take the last such occurrence.
      2. Otherwise, search every known category in the categories list and
         return the one whose LAST occurrence in completion is rightmost
         (i.e. the final decision typically appears near the end).
    Returns '' when nothing matches.
    """
    if not completion or not categories:
        return ''

    # Pattern 1: explicit "类别：xxx" / "分类：xxx" markers, allow markdown bold.
    explicit_re = re.compile(
        r'(?:\*\*\s*)?(?:类别|分类|最终分类|结果|判断|标签)\s*[:：]\s*\*?\*?\s*([^\n\*]+?)\s*(?:\*\*|\*|$|\n)',
        re.MULTILINE,
    )
    matches = explicit_re.findall(completion)
    # Try matches from the last to the first
    for cand in reversed(matches):
        cand = cand.strip().strip('"\'""''「」').strip()
        # Try exact or substring match against known categories
        for cat in categories:
            if cand == cat or cat in cand or cand in cat:
                return cat

    # Pattern 2: rightmost occurrence of any known category in completion.
    best_cat, best_pos = '', -1
    for cat in categories:
        if not cat:
            continue
        pos = completion.rfind(cat)
        if pos > best_pos:
            best_pos = pos
            best_cat = cat
    return best_cat


def _normalize_gt_label(gt: str) -> str:
    """Trim and clean ground-truth label string."""
    if gt is None:
        return ''
    s = str(gt).strip()
    # If gt happens to be the full <explanation>-style output, take the
    # first non-empty line as label.
    if '\n' in s or '<' in s:
        for line in s.splitlines():
            line = line.strip().strip('：:。.,，、 *')
            if line and '<' not in line:
                return line
        return ''
    return s.strip('：:。.,，、 *')


class EventAccuracy(ORM):
    """Reward 1.0 iff the model's predicted category equals the ground-truth label.

    The category list is parsed dynamically from each sample's prompt
    (the "# 类别列表" block in `instruction` / system message), so this
    reward works even when new categories are dynamically introduced.
    """

    def __call__(
        self,
        completions: List[str],
        solution: Optional[List[str]] = None,
        label: Optional[List[str]] = None,
        messages: Optional[List] = None,
        instruction: Optional[List[str]] = None,
        **kwargs,
    ) -> List[float]:
        # Prefer `label` (clean category) over `solution` (also a clean category
        # in our setup, since --columns maps label -> solution).
        gts = label if label is not None else solution
        n = len(completions)
        if gts is None:
            gts = [''] * n
        if messages is None:
            messages = [None] * n
        if instruction is None:
            instruction = [None] * n

        rewards = []
        for i in range(n):
            completion = completions[i] or ''
            gt = _normalize_gt_label(gts[i])
            prompt_text = _get_prompt_text(messages[i], instruction[i])
            categories = _extract_categories_from_text(prompt_text)
            # Make sure GT itself is recognized as a valid category, even if
            # it is dynamically newly introduced and missing from the parsed list.
            if gt and gt not in categories:
                categories.append(gt)
            pred = _find_category_in_completion(completion, categories)
            rewards.append(1.0 if (gt and pred == gt) else 0.0)
        return rewards


class EventFormat(ORM):
    """A lightweight format reward.

    Returns 1.0 when the completion clearly produces a final category that
    belongs to the prompt's category list (regardless of whether the model
    additionally wraps an <explanation> block). Returns 0.0 otherwise.
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

        rewards = []
        for i in range(n):
            completion = completions[i] or ''
            prompt_text = _get_prompt_text(messages[i], instruction[i])
            categories = _extract_categories_from_text(prompt_text)
            pred = _find_category_in_completion(completion, categories)
            rewards.append(1.0 if pred else 0.0)
        return rewards


orms['event_accuracy'] = EventAccuracy
orms['event_format'] = EventFormat
