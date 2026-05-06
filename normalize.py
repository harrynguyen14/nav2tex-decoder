import re


# ── 1. \over → \frac ─────────────────────────────────────────────────────────
_OVER_PATTERN = re.compile(r'\{([^{}]+)\\over\s+([^{}]+)\}')

def _normalize_over(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = _OVER_PATTERN.sub(r'\\frac{\1}{\2}', s)
    return s


# ── 2. \frac variants → \frac ────────────────────────────────────────────────
def _normalize_frac_variants(s: str) -> str:
    s = re.sub(r'\\[tdc]frac\b', r'\\frac', s)
    return s


# ── 3. Font commands ──────────────────────────────────────────────────────────
def _normalize_font_commands(s: str) -> str:
    s = re.sub(r'\{\\bf\s+([^}]*)\}',   r'\\mathbf{\1}',   s)
    s = re.sub(r'\{\\it\s+([^}]*)\}',   r'\\mathit{\1}',   s)
    s = re.sub(r'\{\\rm\s+([^}]*)\}',   r'\\mathrm{\1}',   s)
    s = re.sub(r'\{\\cal\s+([^}]*)\}',  r'\\mathcal{\1}',  s)
    s = re.sub(r'\{\\tt\s+([^}]*)\}',   r'\\mathtt{\1}',   s)
    s = re.sub(r'\{\\sf\s+([^}]*)\}',   r'\\mathsf{\1}',   s)
    s = re.sub(r'\{\\bm\s+([^}]*)\}',   r'\\mathbf{\1}',   s)
    s = re.sub(r'\\boldsymbol\{([^}]*)\}', r'\\mathbf{\1}', s)
    return s


# ── 4. Spacing normalization ──────────────────────────────────────────────────
_SPACE_PAT = re.compile(r'\\[;:!]|\\[ ]|\\thinspace|\\medspace|\\thickspace')

def _normalize_spacing(s: str) -> str:
    s = _SPACE_PAT.sub(r'\\,', s)
    s = re.sub(r'\\qquad\b', r'\\quad', s)
    return s


# ── 5. Symbol aliases ─────────────────────────────────────────────────────────
_SYMBOL_MAP = [
    (r'\\ne\b',               r'\\neq'),
    (r'\\ldots\b',            r'\\dots'),
    (r'\\dotsc\b',            r'\\dots'),
    (r'\\dotsb\b',            r'\\cdots'),
    (r'\\dotso\b',            r'\\dots'),
    (r'\\cdotp\b',            r'\\cdot'),
    (r'\\bullet\b',           r'\\cdot'),
    (r'\\leftarrow\b',        r'\\gets'),
    (r'\\longrightarrow\b',   r'\\to'),
    (r'\\longleftarrow\b',    r'\\gets'),
    (r'\\rightarrow\b',       r'\\to'),
    (r'\\le\b',               r'\\leq'),
    (r'\\ge\b',               r'\\geq'),
    (r'\\varnothing\b',       r'\\emptyset'),
    (r'\\empty\b',            r'\\emptyset'),
    (r'−',               r'-'),       # unicode minus → ASCII minus
    (r'\\operatorname\*',     r'\\operatorname'),
]

def _normalize_symbols(s: str) -> str:
    for pat, repl in _SYMBOL_MAP:
        s = re.sub(pat, repl, s)
    return s


# ── 6. Sub/superscript ordering: X^a_b → X_b^a ───────────────────────────────
_SUP_THEN_SUB = re.compile(
    r'(\{[^{}]*\}|[^{}_^\\]|\\\w+)'
    r'\^(\{[^{}]*\}|[^{}_^ ])'
    r'_(\{[^{}]*\}|[^{}_^ ])'
)

def _swap_sup_sub(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = _SUP_THEN_SUB.sub(r'\1_\3^\2', s)
    return s


# ── 7. Redundant single-char braces in scripts ────────────────────────────────
def _normalize_braces(s: str) -> str:
    # ^{x} → ^x  and  _{x} → _x  for single alphanumeric char only
    s = re.sub(r'\^\{([A-Za-z0-9])\}', r'^\1', s)
    s = re.sub(r'_\{([A-Za-z0-9])\}',  r'_\1',  s)
    return s


# ── 8. Environment aliases ────────────────────────────────────────────────────
def _normalize_environments(s: str) -> str:
    s = re.sub(r'\\begin\{align\*\}',     r'\\begin{align}',    s)
    s = re.sub(r'\\end\{align\*\}',       r'\\end{align}',      s)
    s = re.sub(r'\\begin\{eqnarray\*?\}', r'\\begin{align}',    s)
    s = re.sub(r'\\end\{eqnarray\*?\}',   r'\\end{align}',      s)
    s = re.sub(r'\\begin\{equation\*\}',  r'\\begin{equation}', s)
    s = re.sub(r'\\end\{equation\*\}',    r'\\end{equation}',   s)
    s = re.sub(r'\\begin\{smallmatrix\}', r'\\begin{pmatrix}',  s)
    s = re.sub(r'\\end\{smallmatrix\}',   r'\\end{pmatrix}',    s)
    return s


# ── 9. Multiline whitespace cleanup ───────────────────────────────────────────
def _normalize_multiline(s: str) -> str:
    lines = [l.strip() for l in s.splitlines()]
    result, prev_blank = [], False
    for l in lines:
        blank = l == ''
        if blank and prev_blank:
            continue
        result.append(l)
        prev_blank = blank
    return '\n'.join(result)


def _normalize_array_indent(s: str) -> str:
    def _strip_env(m: re.Match) -> str:
        inner = '\n'.join(l.strip() for l in m.group(2).splitlines())
        return m.group(1) + inner + m.group(3)

    return re.sub(
        r'(\\begin\{(?:array|matrix|pmatrix|bmatrix|vmatrix|Vmatrix|cases|align|gather)\}[^}]*\}?)'
        r'(.*?)'
        r'(\\end\{(?:array|matrix|pmatrix|bmatrix|vmatrix|Vmatrix|cases|align|gather)\})',
        _strip_env,
        s,
        flags=re.DOTALL,
    )


# ── 10. Collapse redundant whitespace ─────────────────────────────────────────
def _normalize_whitespace(s: str) -> str:
    s = re.sub(r'[ \t]+', ' ', s)
    return s.strip()


# ── Public API ────────────────────────────────────────────────────────────────
def normalize(latex: str) -> str:
    try:
        s = latex
        s = _normalize_over(s)
        s = _normalize_frac_variants(s)
        s = _normalize_font_commands(s)
        s = _normalize_environments(s)
        s = _normalize_symbols(s)
        s = _normalize_spacing(s)
        s = _normalize_braces(s)
        s = _swap_sup_sub(s)
        s = _normalize_multiline(s)
        s = _normalize_array_indent(s)
        s = _normalize_whitespace(s)
        return s
    except Exception:
        return latex
