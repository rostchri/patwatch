#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patwatch.py — watch-style Regex-Zähler mit optionalem Kommando, Intervall,
farbigem Header, UTC/Custom-Header, POSIX-Regex-Klassen-Unterstützung.

Pattern-Datei: ID<TAB>LINE_REGEX<TAB>WORD_REGEX(optional)
"""

import sys, re, argparse, subprocess, time, shutil
from collections import deque
from typing import List, Optional, Pattern

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Zeilen zählen & Wörter sammeln nach Regex-Patterns (pro ID).")
    p.add_argument("-p","--patterns", required=True, help="Pattern-Datei: ID<TAB>LINE_REGEX<TAB>WORD_REGEX(optional)")
    p.add_argument("-m","--maxw", type=int, default=10, help="Max. #Head-Wörter (erste N Treffer)")
    p.add_argument("-l","--lastw", type=int, default=10, help="Max. #Tail-Wörter (letzte M Treffer)")
    p.add_argument("--sep", default=" ", help="Trennzeichen zwischen Wörtern (Default: ' ')")
    p.add_argument("--between", default=" ... ", help="Trenner zwischen Head und Tail (Default: ' ... ')")
    p.add_argument("-i","--ignorecase", action="store_true", help="Case-insensitive Matching")
    p.add_argument("--strip-punct", action="store_true", help="Satzzeichen am Wortanfang/-ende entfernen")
    p.add_argument("--fs", default="\\t", help="Feldtrenner in Pattern-Datei (Default: '\\t')")
    # Kommando/Watcher
    p.add_argument("-c","--cmd", help="Shell-Kommando (Pipes erlaubt); dessen STDOUT wird ausgewertet.")
    p.add_argument("-t","--interval", type=float, default=0.0, help="Intervall in Sekunden (watch-artig). 0 = einmalig.")
    p.add_argument("--shell", default="/bin/sh", help="Shell für -c/--cmd (Default: /bin/sh)")
    p.add_argument("--timeout", type=float, default=None, help="Timeout in Sekunden fürs Kommando (optional)")
    p.add_argument("--clear", action="store_true", help="Pro Durchlauf Bildschirm löschen + Header wie 'watch'")
    p.add_argument("--no-warn", action="store_true",
                   help="Unterdrückt STDERR des Kommandos und interne Warnhinweise bei Exit≠0.")
    p.add_argument("--color-header", action="store_true",
                   help="Farbiger Header: dunkles tmux-Grün als Hintergrund, schwarze Schrift.")
    p.add_argument("--utc", action="store_true", help="Timestamp im Header in UTC ausgeben.")
    p.add_argument("--header", default="", help="Custom-Headertext; erscheint oben rechts vor dem Timestamp.")
    return p.parse_args()

def unescape(s: str) -> str:
    return s.encode("utf-8").decode("unicode_escape")

# ---------- POSIX-RegEx Klassen -> Python ----------
_POSIX_RE = re.compile(r"\[\[:(alpha|digit|alnum|space|lower|upper|xdigit|word|punct):\]\]")
_POSIX_MAP = {
    "alpha": r"[A-Za-z]",
    "digit": r"\d",
    "alnum": r"[0-9A-Za-z]",
    "space": r"\s",
    "lower": r"[a-z]",
    "upper": r"[A-Z]",
    "xdigit": r"[0-9A-Fa-f]",
    "word": r"\w",
    "punct": r"[^\w\s]",
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

class PatternRec:
    __slots__ = ("pid","line_re","word_re","count","head","head_count","tail","tail_max")
    def __init__(self, pid: str, line_re: Pattern, word_re: Optional[Pattern],
                 maxw: int, lastw: int):
        self.pid = pid
        self.line_re = line_re
        self.word_re = word_re
        self.count = 0
        self.head: List[str] = []
        self.head_count = 0
        self.tail_max = max(0, lastw)
        self.tail = deque(maxlen=lastw) if lastw > 0 else deque()

def compile_rx(rx: str, flags: int) -> Pattern:
    return re.compile(rx, flags)

def pick_word(line: str, pat: PatternRec) -> str:
    if pat.word_re is not None:
        m = pat.word_re.search(line)
        if not m: return ""
        if m.lastindex:
            for gi in range(1, m.lastindex+1):
                g = m.group(gi)
                if g: return g
            return m.group(1)
        return m.group(0)
    return last_word(line)

def load_patterns(path: str, fs: str, flags: int, maxw: int, lastw: int) -> List[PatternRec]:
    pats: List[PatternRec] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, raw in enumerate(f, 1):
            line = raw.rstrip("\n")
            if not line.strip(): continue
            if line.endswith("\r"): line = line[:-1]
            parts = line.split(fs, 2)
            if len(parts) < 2:
                sys.stderr.write(f"[WARN] Zeile {ln}: erwarte mind. 2 Felder (ID{fs}LINE_REGEX[ {fs}WORD_REGEX]). Übersprungen.\n")
                continue
            pid = parts[0].strip()
            line_rx = posix_to_py(parts[1])
            word_rx = posix_to_py(parts[2]) if len(parts)>=3 and parts[2]!="" else None
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
            pats.append(PatternRec(pid, lre, wre, maxw, lastw))
    return pats

# ---------- Kernverarbeitung ----------
def process_text(input_text: str, patterns: List[PatternRec], maxw: int, sep: str, between: str, strip_p: bool) -> str:
    for pat in patterns:
        pat.count = 0
        pat.head.clear()
        pat.head_count = 0
        if pat.tail_max > 0:
            pat.tail.clear()

    for line in input_text.splitlines():
        for pat in patterns:
            if pat.line_re.search(line):
                pat.count += 1
                w = pick_word(line, pat)
                if strip_p and w:
                    w = strip_punct(w)
                if w:
                    if pat.head_count < maxw:
                        pat.head.append(w)
                        pat.head_count += 1
                    if pat.tail_max > 0:
                        pat.tail.append(w)

    out_lines = []
    for pat in patterns:
        total = pat.count
        h = pat.head_count
        t = len(pat.tail) if pat.tail_max > 0 else 0
        overlap = max(0, h + t - total)

        last_list: List[str] = []
        if t > 0:
            skipped = 0
            for w in pat.tail:
                if skipped < overlap:
                    skipped += 1
                    continue
                last_list.append(w)

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

# ---------- Kommando & Watch ----------
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

def clear_screen():
    sys.stdout.write("\x1b[H\x1b[2J"); sys.stdout.flush()

def render_header(left: str, right: str, color: bool):
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
        # schwarzer Text (30), dunkelgrüner Hintergrund (48;5;22) wie tmux
        sys.stdout.write(f"\x1b[30;48;5;22m{line.ljust(cols)}\x1b[0m\n")
    else:
        sys.stdout.write(line + "\n")

def now_str(use_utc: bool) -> str:
    fmt = "%a %b %d %H:%M:%S %Z %Y"
    return time.strftime(fmt, time.gmtime() if use_utc else time.localtime())

def main():
    args = parse_args()
    fs = unescape(args.fs)
    sep = unescape(args.sep)
    between = unescape(args.between)
    flags = re.IGNORECASE if args.ignorecase else 0

    patterns = load_patterns(args.patterns, fs, flags, args.maxw, args.lastw)
    if not patterns:
        sys.stderr.write("[ERROR] Keine gültigen Patterns geladen.\n"); sys.exit(2)

    def one_run():
        if args.clear:
            clear_screen()
            left = (f"Every {args.interval:.1f}s: {args.cmd}" if args.cmd else "STDIN")
            right_parts = []
            if args.header:
                right_parts.append(args.header)
            right_parts.append(now_str(args.utc))
            right = "  ".join(right_parts)
            render_header(left, right, color=args.color_header)
            print()  # Leerzeile

        text = run_cmd(args.cmd, args.shell, args.timeout, args.no_warn) if args.cmd else sys.stdin.read()
        sys.stdout.write(process_text(text, patterns, args.maxw, sep, between, args.strip_punct))
        sys.stdout.flush()

    if args.interval and args.interval > 0:
        if not args.cmd:
            sys.stderr.write("[ERROR] --interval erfordert --cmd (STDIN kann nicht periodisch gelesen werden).\n")
            sys.exit(2)
        try:
            while True:
                one_run()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass
    else:
        one_run()

if __name__ == "__main__":
    main()
