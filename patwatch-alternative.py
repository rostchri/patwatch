#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patwatch-alternative.py — Regex-Zähler mit watch-Style-Header, Anti-Flackern,
Template (Spalte 4), Transform-Pipeline (Spalte 5 inkl. Backrefs als Quelle),
Live-Toggle 'a' (Alternativansicht), Aux-Kommando, Laufzeitmessung
und optionale Farbchips-Ausgabe (--color).

Pattern-TSV (Tab-getrennt):
  ID<TAB>LINE_REGEX<TAB>WORD_REGEX(optional)<TAB>TEMPLATE(optional)<TAB>TRANSFORMS(optional)

Extras:
- Zeilen in der Pattern-Datei, die (nach optionalen Spaces) mit '#' beginnen, werden ignoriert.
- Header zeigt rechts Timestamp (+ optional Custom-Header) und die reine Verarbeitungszeit in ms: "[123ms]".
- Taste 'a' schaltet zwischen Normalansicht (Head/Tail) und Alternativansicht (Breiten-optimiert) um.
- --color erzeugt farbige Wort-"Chips" mit ≥120 stabilen, gut lesbaren ANSI-256 Farbkombis (FG/BG),
  mit WCAG-orientierter Kontrastwahl (nie Schwarz auf starkem Rot/Blau).
- Farben in der Normalansicht entsprechen den Farben der Alternativansicht:
  Die Farbauswahl basiert auf dem Alternativ-Wort (Spalte 5); fehlt dieses, auf dem Originalwort.
"""

import sys, re, argparse, subprocess, time, shutil, os, select
from collections import deque
from typing import List, Optional, Pattern, Tuple

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Zeilen zählen & Wörter sammeln nach Regex-Patterns (pro ID).")
    p.add_argument("-p","--patterns", required=True,
                   help="Pattern-Datei: ID<TAB>LINE_REGEX<TAB>WORD_REGEX(optional)<TAB>TEMPLATE(optional)<TAB>TRANSFORMS(optional)")
    p.add_argument("-m","--maxw", type=int, default=10, help="Max. #Head-Wörter (erste N Treffer)")
    p.add_argument("-l","--lastw", type=int, default=10, help="Max. #Tail-Wörter (letzte M Treffer)")
    p.add_argument("--sep", default=" ", help="Trennzeichen zwischen Wörtern (Default: ' ')")
    p.add_argument("--between", default=" ... ", help="Trenner zwischen Head/Tail bzw. Mitte in Alternativansicht")
    p.add_argument("-i","--ignorecase", action="store_true", help="Case-insensitive Matching")
    p.add_argument("--strip-punct", action="store_true", help="Satzzeichen am Wortanfang/-ende entfernen")
    p.add_argument("--fs", default="\\t", help="Feldtrenner in Pattern-Datei (Default: '\\t')")
    p.add_argument("--cg-sep", default="", help="Trenner beim Zusammenfügen mehrerer Capture-Gruppen (wenn kein TEMPLATE)")

    # Hauptkommando / Watch
    p.add_argument("-c","--cmd", help="Shell-Kommando (Pipes erlaubt); dessen STDOUT wird ausgewertet.")
    p.add_argument("-t","--interval", type=float, default=0.0, help="Intervall in Sekunden (watch-artig). 0 = einmalig.")
    p.add_argument("--shell", default="/bin/sh", help="Shell für -c/--cmd und --auxcmd (Default: /bin/sh)")
    p.add_argument("--timeout", type=float, default=None, help="Timeout in Sekunden fürs Hauptkommando (optional)")
    p.add_argument("--clear", action="store_true", help="Pro Durchlauf Bildschirm löschen + Header wie 'watch'")
    p.add_argument("--no-warn", action="store_true",
                   help="Unterdrückt STDERR der Kommandos und interne Warnhinweise bei Exit≠0.")
    p.add_argument("--color-header", action="store_true",
                   help="Farbiger Header: tmux-dunkelgrün (BG 48;5;22) + schwarzer Text (30).")

    # Header-Extras
    p.add_argument("--utc", action="store_true", help="Timestamp im Header in UTC ausgeben.")
    p.add_argument("--header", default="", help="Custom-Headertext; erscheint oben rechts vor dem Timestamp.")

    # Aux-Kommando
    p.add_argument("-a","--auxcmd", help="Zweites Shell-Kommando; dessen STDOUT wird angehängt.")
    p.add_argument("--aux-sep", default="###",
                   help="Trennerzeile vor/zwischen Haupt- und Aux-Output (Default: '###'). Escape wie '\\n' erlaubt.")
    p.add_argument("--aux-timeout", type=float, default=None,
                   help="Eigenes Timeout (Sekunden) für --auxcmd. Default: --timeout.")
    p.add_argument("--aux-before", action="store_true",
                   help="Aux-Block vor dem Hauptblock ausgeben.")

    # Farben
    p.add_argument("--color", action="store_true",
                   help="Farbige Wort-Chips (≥120 ANSI-256 Farbkombis). Ignoriert --sep; gilt in Normal- und Alt-Ansicht.")
    return p.parse_args()

def unescape(s: str) -> str:
    return s.encode("utf-8").decode("unicode_escape")

# ---------- POSIX-RegEx Klassen -> Python ----------
_POSIX_RE = re.compile(r"\[\[:(alpha|digit|alnum|space|lower|upper|xdigit|word|punct):\]\]")
_POSIX_MAP = {
    "alpha": r"[A-Za-z]", "digit": r"\d", "alnum": r"[0-9A-Za-z]",
    "space": r"\s", "lower": r"[a-z]", "upper": r"[A-Z]",
    "xdigit": r"[0-9A-Fa-f]", "word": r"\w", "punct": r"[^\w\s]",
}
def posix_to_py(rx: str) -> str:
    return _POSIX_RE.sub(lambda m: _POSIX_MAP[m.group(1)], rx)

# ---------- Wort-Extraktion & Helpers ----------
def last_word(line: str) -> str:
    line = line.rstrip()
    if not line: return ""
    parts = re.split(r"\s+", line)
    return parts[-1] if parts else ""

def strip_punct(word: str) -> str:
    return re.sub(r'^[^\w\s]+|[^\w\s]+$', "", word)

def apply_template(tmpl: str, m: Optional[re.Match]) -> str:
    """Backrefs im Template via Match m ersetzen. Unterstützt: \1..\99, \g<name>, \\ \t \n \r.
       Wenn m=None → Backrefs werden zu leerem String; Escapes bleiben wirksam.
    """
    if m is None:
        s = tmpl.replace("\\t", "\t").replace("\\n", "\n").replace("\\r", "\r").replace("\\\\", "\\")
        return re.sub(r'\\(?:g<[^>]+>|\d+)', "", s)

    out = []; i = 0; s = tmpl; L = len(s)
    while i < L:
        ch = s[i]
        if ch != '\\':
            out.append(ch); i += 1; continue
        i += 1
        if i >= L: out.append('\\'); break
        c = s[i]
        if c.isdigit():
            j = i
            while j < L and s[j].isdigit(): j += 1
            idx = int(s[i:j]) if j > i else None
            try:
                out.append(m.group(idx) or "" if idx is not None else "")
            except IndexError:
                out.append("")
            i = j; continue
        if c == 'g' and i + 1 < L and s[i+1] == '<':
            k = s.find('>', i+2)
            if k != -1:
                name = s[i+2:k]
                try:
                    out.append(m.group(name) or "")
                except Exception:
                    out.append("")
                i = k + 1; continue
            out.append('\\g'); i += 1; continue
        if c == 't': out.append('\t'); i += 1; continue
        if c == 'n': out.append('\n'); i += 1; continue
        if c == 'r': out.append('\r'); i += 1; continue
        if c == '\\': out.append('\\'); i += 1; continue
        out.append(c); i += 1
    return "".join(out)

# ---------- Transform-Pipeline ----------
def parse_pipeline(s: str):
    """Splitte Pipeline an unescaped '|'. Erhalte Backslashes; behandle nur '\|' und '\\' speziell."""
    if not s: return []
    tokens, cur = [], []
    i, L = 0, len(s)
    while i < L:
        ch = s[i]
        if ch == '\\':
            if i + 1 < L:
                nxt = s[i+1]
                if nxt == '|':
                    cur.append('|'); i += 2; continue
                elif nxt == '\\':
                    cur.append('\\'); i += 2; continue
                else:
                    cur.append('\\'); i += 1; continue
            else:
                cur.append('\\'); i += 1; continue
        elif ch == '|':
            tokens.append(''.join(cur).strip()); cur = []; i += 1; continue
        else:
            cur.append(ch); i += 1
    if cur:
        tokens.append(''.join(cur).strip())
    return [t for t in tokens if t]

_CALL_RE = re.compile(r'^([a-zA-Z_]\w*)\s*(?:\((.*)\))?$')

def split_call(token: str):
    m = _CALL_RE.match(token)
    if not m: return None, token  # kein Funktionsaufruf → Template/Backref-Token
    name, args = m.group(1), (m.group(2) or "")
    out, cur, q, esc = [], [], None, False
    for ch in args:
        if esc: cur.append(ch); esc=False; continue
        if ch == '\\': esc=True; continue
        if q:
            if ch == q: q=None
            else: cur.append(ch)
        else:
            if ch in ("'", '"'): q=ch
            elif ch == ',': out.append(''.join(cur).strip()); cur=[]
            else: cur.append(ch)
    if cur: out.append(''.join(cur).strip())
    out = [a[1:-1] if (len(a)>=2 and a[0]==a[-1] and a[0] in "'\"") else a for a in out]
    return name.lower(), out

def apply_pipeline(word: str, pipeline: str, m: Optional[re.Match]) -> str:
    funcs = {
        # String-Case
        "upper": lambda s: s.upper(),
        "lower": lambda s: s.lower(),
        "title": lambda s: s.title(),
        "swapcase": lambda s: s.swapcase(),
        # Slicing
        "first": lambda s,n="1": s[:int(n)],
        "last":  lambda s,n="1": s[-int(n):] if s else s,
        "slice": lambda s,a="",b="": s[(int(a) if a!="" else None):(int(b) if b!="" else None)],
        "subst": lambda s,a="",b="": s[(int(a) if a!="" else None):(int(b) if b!="" else None)],  # Alias
        # Trim/Whitespace
        "strip": lambda s,chs="": s.strip(chs) if chs else s.strip(),
        "lstrip": lambda s,chs="": s.lstrip(chs) if chs else s.lstrip(),
        "rstrip": lambda s,chs="": s.rstrip(chs) if chs else s.rstrip(),
        "collapse_ws": lambda s: " ".join(s.split()),
        # Ersetzen
        "replace_str": lambda s,a,b: s.replace(a,b),
        "replace": lambda s,rx,repl,flag="": re.sub(rx, repl, s, flags=(re.I if ('i' in flag.lower()) else 0)),
        "tr": lambda s,src,dst: s.translate(str.maketrans(src, dst)),
        # Split/Extract
        "split": lambda s,delim,idx="0": (s.split(delim)[int(idx)] if delim in s and len(s.split(delim))>int(idx) else ""),
        "rsplit": lambda s,delim,idx="0": (s.rsplit(delim)[int(idx)] if delim in s and len(s.rsplit(delim))>int(idx) else ""),
        "rextract": lambda s,rx,grp="1": (re.search(rx,s).group(int(grp)) if re.search(rx,s) else ""),
        # Padding/Format
        "padleft": lambda s,w,ch=" ": s.rjust(int(w), ch[:1] if ch else " "),
        "padright": lambda s,w,ch=" ": s.ljust(int(w), ch[:1] if ch else " "),
        "zfill": lambda s,w: s.zfill(int(w)),
        "ensure_prefix": lambda s,p: s if s.startswith(p) else (p+s),
        "ensure_suffix": lambda s,p: s if s.endswith(p) else (s+p),
        # Zahlen (best effort)
        "int": lambda s: str(int(float(s))) if s.strip() else s,
        "float": lambda s: str(float(s)) if s.strip() else s,
        "round": lambda s,d="0": (("{0:." + str(int(d)) + "f}").format(round(float(s), int(d)))) if s.strip() else s,
        # Mit Templates im Lauf setzen/anhängen
        "set": lambda s,expr: apply_template(expr, m),
        "append": lambda s,expr: s + apply_template(expr, m),
        "prepend": lambda s,expr: apply_template(expr, m) + s,
        "concat": lambda s,expr: s + apply_template(expr, m),
    }

    try:
        tokens = parse_pipeline(pipeline)
        for tok in tokens:
            call = split_call(tok)
            if call[0] is None:
                # Kein Funktionsaufruf → behandle als Template/Backref-Token (ersetzen)
                word = apply_template(call[1], m)
                continue
            name, args = call
            f = funcs.get(name)
            if not f:
                continue
            if name in ("set","append","prepend","concat") and args:
                word = f(word, args[0])  # apply_template passiert in Func
            else:
                word = f(word, *args)
    except Exception:
        pass
    return word

# ---------- Datenstruktur ----------
class PatternRec:
    __slots__ = ("pid","line_re","word_re","tmpl","transforms",
                 "count","head","head_keys","head_count","tail","tail_keys","tail_max","alts")
    def __init__(self, pid: str, line_re: Pattern, word_re: Optional[Pattern],
                 tmpl: str, transforms: str, maxw: int, lastw: int):
        self.pid = pid
        self.line_re = line_re
        self.word_re = word_re
        self.tmpl = tmpl
        self.transforms = transforms  # 5. Spalte (Pipeline)
        self.count = 0
        self.head: List[str] = []
        self.head_keys: List[str] = []   # Farb-Key je Head-Wort (Altwort dominiert)
        self.head_count = 0
        self.tail_max = max(0, lastw)
        self.tail = deque(maxlen=lastw) if lastw > 0 else deque()
        self.tail_keys = deque(maxlen=lastw) if lastw > 0 else deque()  # Farb-Key je Tail-Wort
        self.alts: List[str] = []  # Historie alternativer Wörter (für Alt-Ansicht)

def compile_rx(rx: str, flags: int) -> Pattern:
    return re.compile(rx, flags)

def extract_word_and_match(line: str, pat: PatternRec, cg_sep: str) -> Tuple[str, Optional[re.Match]]:
    """Gibt (Wort, Matchobjekt) zurück. Matchobjekt wird für Transforms/Backrefs in Spalte 5 genutzt."""
    if pat.word_re is not None:
        m = pat.word_re.search(line)
        if not m:
            return "", None
        if pat.tmpl:
            return apply_template(pat.tmpl, m), m
        if m.lastindex:
            parts = [g for g in m.groups() if g]
            return (cg_sep.join(parts) if parts else m.group(0)), m
        return m.group(0), m
    # ohne WORD_REGEX
    return last_word(line), None

def load_patterns(path: str, fs: str, flags: int, maxw: int, lastw: int) -> List[PatternRec]:
    pats: List[PatternRec] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, raw in enumerate(f, 1):
            line = raw.rstrip("\n")
            if not line.strip(): continue
            if line.endswith("\r"): line = line[:-1]
            if line.lstrip().startswith("#"):  # Kommentarzeilen ignorieren
                continue
            parts = line.split(fs, 4)  # bis zu 5 Felder
            if len(parts) < 2:
                sys.stderr.write(f"[WARN] Zeile {ln}: erwarte mind. 2 Felder (ID{fs}LINE_REGEX[ {fs}WORD_REGEX[ {fs}TEMPLATE[ {fs}TRANSFORMS ]]]). Übersprungen.\n")
                continue
            pid = parts[0].strip()
            line_rx = posix_to_py(parts[1])
            word_rx = posix_to_py(parts[2]) if len(parts)>=3 and parts[2] != "" else None
            tmpl = parts[3] if len(parts) >= 4 else ""
            transforms = parts[4] if len(parts) >= 5 else ""
            if not pid or not line_rx:
                sys.stderr.write(f"[WARN] Zeile {ln}: leere ID oder LINE_REGEX. Übersprungen.\n")
                continue
            try:
                lre = compile_rx(line_rx, flags)
            except re.error as e:
                sys.stderr.write(f"[WARN] Zeile {ln}: ungültige LINE_REGEX '{parts[1]}': {e}. Übersprungen.\n")
                continue
            wre = None
            if word_rx is not None:
                try: wre = compile_rx(word_rx, flags)
                except re.error as e:
                    sys.stderr.write(f"[WARN] Zeile {ln}: ungültige WORD_REGEX '{parts[2]}': {e}. WORD_REGEX ignoriert.\n")
            pats.append(PatternRec(pid, lre, wre, tmpl, transforms, maxw, lastw))
    return pats

# ---------- Kernverarbeitung ----------
def process_text(input_text: str, patterns: List[PatternRec], maxw: int, sep: str, between: str,
                 strip_p: bool, cg_sep: str) -> str:
    # reset
    for pat in patterns:
        pat.count = 0
        pat.head.clear()
        pat.head_keys.clear()
        pat.head_count = 0
        pat.alts.clear()
        if pat.tail_max > 0:
            pat.tail.clear()
            pat.tail_keys.clear()

    for line in input_text.splitlines():
        for pat in patterns:
            if pat.line_re.search(line):
                pat.count += 1
                w, m = extract_word_and_match(line, pat, cg_sep)
                if strip_p and w: w = strip_punct(w)
                # Transformiertes Alternativwort (Spalte 5; darf Backrefs als Quelle nutzen)
                alt = apply_pipeline(w, pat.transforms, m) if pat.transforms else w
                pat.alts.append(alt)
                # Normal-Head/Tail
                if w:
                    if pat.head_count < maxw:
                        pat.head.append(w)
                        pat.head_keys.append(alt if alt else w)  # Farb-Key: Altwort dominiert
                        pat.head_count += 1
                    if pat.tail_max > 0:
                        pat.tail.append(w)
                        pat.tail_keys.append(alt if alt else w)  # parallel zu tail

    # Normalansicht (Plain; wird bei --color u. U. nicht genutzt)
    out_lines = []
    for pat in patterns:
        total = pat.count
        h = pat.head_count
        t = len(pat.tail) if pat.tail_max > 0 else 0
        overlap = max(0, h + t - total)

        last_list: List[str] = []
        last_keys: List[str] = []
        if t > 0:
            skipped = 0
            # tail und tail_keys parallel behandeln
            for w, k in zip(list(pat.tail), list(pat.tail_keys)):
                if skipped < overlap:
                    skipped += 1
                    continue
                last_list.append(w)
                last_keys.append(k)

        shown_head = h
        shown_tail = len(last_list)
        gap_exists = (total > shown_head + shown_tail)

        if pat.head and last_list:
            joiner = between if gap_exists else sep
            words = sep.join(pat.head) + joiner + sep.join(last_list)
        elif pat.head:
            words = sep.join(pat.head)
        else:
            words = sep.join(last_list)

        out_lines.append(f"{pat.pid}\t{total}\t{words}")
    return "\n".join(out_lines) + ("\n" if out_lines else "")

# ---------- Farben (ANSI 256) mit Kontrast-Heuristik ----------
def _xterm_rgb(code: int):
    if 16 <= code <= 231:  # 6x6x6 farbwürfel
        i = code - 16
        r = i // 36
        g = (i % 36) // 6
        b = i % 6
        conv = lambda v: 0 if v == 0 else 95 + (v - 1) * 40
        return (conv(r), conv(g), conv(b))
    if 232 <= code <= 255:  # grau
        v = 8 + 10 * (code - 232)
        return (v, v, v)
    return (255, 255, 255)

def _srgb_to_linear(c: float) -> float:
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

def _rel_lum_rgb(r: int, g: int, b: int) -> float:
    R = _srgb_to_linear(r); G = _srgb_to_linear(g); B = _srgb_to_linear(b)
    return 0.2126 * R + 0.7152 * G + 0.0722 * B

def _contrast(l1: float, l2: float) -> float:
    if l1 < l2: l1, l2 = l2, l1
    return (l1 + 0.05) / (l2 + 0.05)

def _best_fg_for_bg(r: int, g: int, b: int) -> int:
    """Wähle 15 (weiß) oder 16 (schwarz) – höchster Kontrast; nie Schwarz auf starkem Rot/Blau."""
    lum = _rel_lum_rgb(r, g, b)
    cr_black = _contrast(lum, 0.0)
    cr_white = _contrast(1.0, lum)
    dominant_red  = (r > g * 1.25) and (r > b * 1.25)
    dominant_blue = (b > g * 1.25) and (b > r * 1.25)
    if dominant_red or dominant_blue:
        return 15  # weiß
    return 15 if cr_white >= cr_black else 16

def build_palette() -> list[tuple[int, int]]:
    """Erzeuge (fg,bg)-Paare mit gutem Kontrast; mind. 120 Stück."""
    pairs = []
    for code in range(16, 232):
        r, g, b = _xterm_rgb(code)
        y8 = 0.2126 * r + 0.7152 * g + 0.0722 * b
        if y8 < 25 or y8 > 245:
            continue
        fg = _best_fg_for_bg(r, g, b)
        pairs.append((fg, code))
    for code in range(238, 247):
        r, g, b = _xterm_rgb(code)
        fg = _best_fg_for_bg(r, g, b)
        pairs.append((fg, code))
    seen, uniq = set(), []
    for fg, bg in pairs:
        if (fg, bg) not in seen:
            uniq.append((fg, bg)); seen.add((fg, bg))
    if len(uniq) < 120:
        for code in range(16, 232):
            r, g, b = _xterm_rgb(code)
            fg = _best_fg_for_bg(r, g, b)
            if (fg, code) not in seen:
                uniq.append((fg, code)); seen.add((fg, code))
            if len(uniq) >= 120:
                break
    return uniq[:240]

_PALETTE = build_palette()

class _ColorAllocator:
    """Sequenzieller Allokator: unterschiedliche Keys → unterschiedliche Farben (bis Palette erschöpft)."""
    def __init__(self, palette):
        self.palette = list(palette)
        self.map = {}       # key -> palette index
        self.next_idx = 0
    def pair_for(self, key: str):
        """Gibt (fg,bg) für den Key (typisch: Altwort) zurück; neues Paar bei Bedarf."""
        if key not in self.map:
            self.map[key] = self.next_idx % len(self.palette)
            self.next_idx += 1
        return self.palette[self.map[key]]

_COLOR_ALLOC = _ColorAllocator(_PALETTE)

def _color_chip_key(word: str, key: Optional[str] = None) -> str:
    """Farbiges Chip-Rendering; Farbe anhand 'key' (z. B. Altwort) wählen."""
    if not word:
        return ""
    fg, bg = _COLOR_ALLOC.pair_for(key if key is not None else word)
    return f"\x1b[38;5;{fg};48;5;{bg}m{word}\x1b[0m"

def _color_chip(word: str) -> str:
    if not word:
        return ""
    # Standard: Farbe aus dem Wort selbst ableiten (Alt-Ansicht nutzt Altwörter direkt)
    fg, bg = _COLOR_ALLOC.pair_for(word)
    return f"\x1b[38;5;{fg};48;5;{bg}m{word}\x1b[0m"

def colorize_join(words: list[str]) -> str:
    chips = [_color_chip(w) for w in words if w]
    # Im Color-Mode ohne sichtbaren Separator zwischen Chips
    return "".join(chips)

def colorize_join_with_keys(words: list[str], keys: list[str]) -> str:
    """Wie colorize_join, aber Farbe wird aus keys[i] abgeleitet (Alt dominiert)."""
    chips = []
    for i, w in enumerate(words):
        if not w:
            continue
        key = keys[i] if i < len(keys) and keys[i] else w
        chips.append(_color_chip_key(w, key))
    return "".join(chips)

# ---------- Rendering (Normal & Alt) ----------
def render_normal_view(patterns: List[PatternRec], sep: str, between: str, use_color: bool) -> str:
    out_lines = []
    for pat in patterns:
        total = pat.count
        h = pat.head_count
        t = len(pat.tail) if pat.tail_max > 0 else 0
        overlap = max(0, h + t - total)

        last_list: List[str] = []
        last_keys: List[str] = []
        if t > 0:
            skipped = 0
            # tail und tail_keys parallel behandeln
            for w, k in zip(list(pat.tail), list(pat.tail_keys)):
                if skipped < overlap:
                    skipped += 1
                    continue
                last_list.append(w)
                last_keys.append(k)

        shown_head = h
        shown_tail = len(last_list)
        gap_exists = (total > shown_head + shown_tail)

        if use_color:
            if pat.head and last_list:
                if gap_exists:
                    words = colorize_join_with_keys(pat.head, pat.head_keys) + between + \
                            colorize_join_with_keys(last_list, last_keys)
                else:
                    words = colorize_join_with_keys(pat.head + last_list, pat.head_keys + last_keys)
            elif pat.head:
                words = colorize_join_with_keys(pat.head, pat.head_keys)
            else:
                words = colorize_join_with_keys(last_list, last_keys)
        else:
            if pat.head and last_list:
                joiner = between if gap_exists else sep
                words = sep.join(pat.head) + joiner + sep.join(last_list)
            elif pat.head:
                words = sep.join(pat.head)
            else:
                words = sep.join(last_list)

        out_lines.append(f"{pat.pid}\t{total}\t{words}")
    return "\n".join(out_lines) + ("\n" if out_lines else "")

def render_alt_view(patterns: List[PatternRec], sep: str, between: str, use_color: bool) -> str:
    cols = shutil.get_terminal_size(fallback=(80, 24)).columns
    # Bei --color werden Chips ohne sichtbaren Separator gerendert → sep_len = 0
    sep_len = (0 if use_color else len(sep))
    between_len = len(between)

    def j_plain(ws): return sep.join(ws) if ws else ""
    def j_color(ws): return colorize_join(ws)

    lines = []
    for pat in patterns:
        prefix = f"{pat.pid}\t{pat.count}\t"
        avail = max(0, cols - len(prefix))
        words = pat.alts
        if not words or avail <= 0:
            lines.append(prefix + "\n"); continue

        # 1) Passt alles? (kein between)
        full = (j_color(words) if use_color else j_plain(words))
        if len(full) <= avail:
            lines.append(prefix + full + ("\n" if not full.endswith("\n") else ""))
            continue

        # 2) Overflow → Mitte mit between auslassen (nie am Rand)
        n = len(words)
        pref_len = [0]*(n+1)
        for i in range(1, n+1):
            pref_len[i] = pref_len[i-1] + len(words[i-1]) + (0 if i-1==0 else sep_len)
        suff_len = [0]*(n+1)
        for j in range(1, n+1):
            suff_len[j] = suff_len[j-1] + len(words[n-j]) + (0 if j-1==0 else sep_len)

        best = None  # (shown, -|balance|, used, i, j)
        for i in range(1, n):
            left_len = pref_len[i]
            if left_len + between_len > avail: break
            rem = avail - left_len - between_len
            j_max = 0
            for j in range(1, n - i + 1):
                if suff_len[j] <= rem: j_max = j
                else: break
            if j_max <= 0 or i + j_max >= n: continue
            used = left_len + between_len + suff_len[j_max]
            cand = (i + j_max, -abs(i - j_max), used, i, j_max)
            if (best is None) or (cand > best): best = cand

        if best:
            _,_,_, i, j = best
            left = words[:i]; right = words[n-j:]
            left_s  = (j_color(left)  if use_color else j_plain(left))
            right_s = (j_color(right) if use_color else j_plain(right))
            content = left_s + between + right_s
            lines.append(prefix + content + ("\n" if not content.endswith("\n") else ""))
            continue

        # 3) Fallback: selbst "1 Wort + between + 1 Wort" passt nicht → kein between
        i_max = 0
        for i in range(1, n+1):
            if pref_len[i] <= avail: i_max = i
            else: break
        j_max = 0
        for j in range(1, n+1):
            if suff_len[j] <= avail: j_max = j
            else: break
        left_only  = (j_color(words[:i_max]) if use_color else j_plain(words[:i_max]))
        right_only = (j_color(words[n-j_max:]) if use_color else j_plain(words[n-j_max:]))
        content = left_only if len(left_only) >= len(right_only) else right_only
        lines.append(prefix + (content if content else "") + ("\n" if content else "\n"))
    return "".join(lines)

# ---------- Kommando & Header/Watch ----------
def run_cmd(cmd: str, shell_path: str, timeout: Optional[float], no_warn: bool) -> str:
    completed = subprocess.run(
        [shell_path, "-c", cmd],
        check=False,
        stdout=subprocess.PIPE,
        stderr=(subprocess.DEVNULL if no_warn else subprocess.PIPE),
        text=True,
        timeout=timeout,
    )
    if (not no_warn) and completed.returncode != 0 and completed.stderr:
        sys.stderr.write(f"[WARN] Kommando Exit {completed.returncode}: {completed.stderr}\n")
    return completed.stdout or ""

def now_str(use_utc: bool) -> str:
    fmt = "%a %b %d %H:%M:%S %Z %Y"
    return time.strftime(fmt, time.gmtime() if use_utc else time.localtime())

def build_header_line(left: str, right: str, color: bool) -> str:
    cols = shutil.get_terminal_size(fallback=(80, 24)).columns
    if len(left) + 1 + len(right) <= cols:
        spaces = cols - len(left) - len(right)
        line = left + (" " * spaces) + right
    else:
        keep_left = max(0, cols - len(right) - 1)
        left = left[:keep_left]
        max_right = max(0, cols - len(left) - 1)
        right = right[-max_right:] if max_right > 0 else ""
        sep = " " if (cols - len(left) - len(right)) > 0 else ""
        line = left + sep + right
    if color:
        # schwarzer Text (30) auf tmux-dunkelgrünem Hintergrund (48;5;22)
        return f"\x1b[30;48;5;22m{line.ljust(cols)}\x1b[0m\n"
    return line + "\n"

# ---------- Tastatur-Poll (für 'a' Toggle) ----------
class KeyPoller:
    def __init__(self):
        self.enabled = sys.stdin.isatty()
        self.old = None
        self.win = os.name == "nt"
        if self.enabled and not self.win:
            import termios, tty
            self.termios = termios; self.tty = tty
    def __enter__(self):
        if self.enabled and not self.win:
            self.old = self.termios.tcgetattr(sys.stdin.fileno())
            self.tty.setcbreak(sys.stdin.fileno())
        return self
    def __exit__(self, exc_type, exc, tb):
        if self.enabled and not self.win and self.old:
            self.termios.tcsetattr(sys.stdin.fileno(), self.termios.TCSADRAIN, self.old)
    def poll(self):
        if not self.enabled: return ""
        if self.win:
            try:
                import msvcrt
                s = ""
                while msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    s += ch
                return s
            except Exception:
                return ""
        else:
            r,_,_ = select.select([sys.stdin], [], [], 0)
            if r:
                try:
                    return sys.stdin.read(1)
                except Exception:
                    return ""
            return ""

def main():
    args = parse_args()
    fs = unescape(args.fs)
    sep = unescape(args.sep)
    between = unescape(args.between)
    cg_sep = unescape(args.cg_sep)
    aux_sep = unescape(args.aux_sep)
    flags = re.IGNORECASE if args.ignorecase else 0

    patterns = load_patterns(args.patterns, fs, flags, args.maxw, args.lastw)
    if not patterns:
        sys.stderr.write("[ERROR] Keine gültigen Patterns geladen.\n"); sys.exit(2)

    alt_mode = False  # Start im Normalmodus

    def one_frame():
        nonlocal alt_mode

        # --- 0) Hauptkommando (nicht in Laufzeitmessung enthalten) ---
        text = run_cmd(args.cmd, args.shell, args.timeout, args.no_warn) if args.cmd else sys.stdin.read()

        # --- 1) Unsere Verarbeitungszeit starten ---
        t_start = time.perf_counter()

        # Normalansicht vorbereiten (inkl. Match/Transform)
        normal_out_plain = process_text(text, patterns, args.maxw, sep, between, args.strip_punct, cg_sep)

        # Inhalt je Modus (mit/ohne Farbe)
        if alt_mode:
            content = render_alt_view(patterns, sep, between, use_color=args.color)
        else:
            content = render_normal_view(patterns, sep, between, use_color=args.color) \
                      if args.color else normal_out_plain

        # Zwischensumme unserer Laufzeit bis hier
        t_proc = time.perf_counter() - t_start

        # --- 2) Aux-Kommando (wird NICHT mitgezählt) ---
        frame_body = content
        if args.auxcmd:
            aux_to = args.aux_timeout if args.aux_timeout is not None else args.timeout
            aux_out = run_cmd(args.auxcmd, args.shell, aux_to, args.no_warn)
            t_after_aux = time.perf_counter()
            sep_line = aux_sep + ("" if aux_sep.endswith("\n") else "\n")
            aux_block = sep_line + (aux_out if aux_out.endswith("\n") else aux_out + "\n")
            frame_body = (aux_block + content) if args.aux_before else (content + aux_block)
            t_proc += (time.perf_counter() - t_after_aux)  # nur das Zusammenbauen addieren

        # --- 3) Header bauen (mit ms) & atomar ausgeben ---
        frame = frame_body
        if args.clear:
            left = (f"Every {args.interval:.1f}s: {args.cmd}" if args.cmd else "STDIN")
            mode_tag = " ALT" if alt_mode else ""
            ts = now_str(args.utc)
            ms = int(round(t_proc * 1000.0))
            right_parts = []
            if args.header or mode_tag:
                right_parts.append((args.header + mode_tag).strip())
            right_parts.append(f"{ts} [{ms}ms]")
            right = "  ".join([p for p in right_parts if p])
            header = build_header_line(left, right, color=args.color_header) + "\n"
            frame = "\x1b[H\x1b[2J" + header + frame_body

        sys.stdout.write(frame); sys.stdout.flush()

    if args.interval and args.interval > 0:
        if not args.cmd:
            sys.stderr.write("[ERROR] --interval erfordert --cmd (STDIN kann nicht periodisch gelesen werden).\n")
            sys.exit(2)
        try:
            with KeyPoller() as kp:
                while True:
                    keys = kp.poll()
                    if keys and 'a' in keys: alt_mode = not alt_mode
                    one_frame()
                    time.sleep(args.interval)
        except KeyboardInterrupt:
            pass
    else:
        one_frame()

if __name__ == "__main__":
    main()

