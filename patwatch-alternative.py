#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patwatch.py — Regex-Zähler mit watch-Style-Header, Anti-Flackern,
Transform-Pipeline (5. Spalte), Live-Toggle 'a' für Alternativansicht,
und Aux-Kommando.

Pattern-TSV:
  ID<TAB>LINE_REGEX<TAB>WORD_REGEX(optional)<TAB>TEMPLATE(optional)<TAB>TRANSFORMS(optional)

Wortbildung pro Match:
- Wenn WORD_REGEX vorhanden:
    * Mit TEMPLATE (Spalte 4): Backrefs \1..\10 und \g<name> + Escapes \\ \t \n \r.
    * Ohne TEMPLATE: alle Capture-Groups (1..n) mit --cg-sep zusammen; wenn keine Gruppen → group(0).
- Wenn keine WORD_REGEX: letztes Wort der Zeile.

Pattern-Template-Beispiele:

- Output-Zeilen: 
   - `user=john.doe@example.com` 
- Pattern: `Mail<TAB>user=<TAB>user=([^@\s]+)(@([^\s]+))?<TAB>\1 \2`
- Output-Zeilen:
   - `... SL 42 ... ID X9 ... NA 7 ... AL 3 ...`
- Pattern: `Flex<TAB>^<TAB>(?=.*\bID (?P<id>\S+))(?=.*\bNA (?P<na>\d+))(?=.*\bAL (?P<al>\d+))(?=.*\bSL (?P<sl>\d+))<TAB>\g<sl>@\g<id> [NA=\g<na>,AL=\g<al>]`
- Output-Zeilen: 
   - `USER=alice (prod)`
   - `USER=bob`
- Pattern: `User<TAB>^USER=<TAB>USER=(\S+)( (\((\w+)\)))?<TAB>\1\2` (Gruppe \2 umfasst inklusive Leerzeichen den ganzen optionalen Block (Space + „(…)“). Ist er nicht vorhanden, wird \2 leer → keine überflüssigen Klammern/Leerzeichen.)
- Wissenswertes zur Template-Spalte:
  - **Backrefs:** `\1..\9..\10` (Achtung: `\10` ist **Gruppe 10**, nicht `\1` + „0“), `\g<name>` für named groups.
  - **Escapes im Template:** `\\`, `\t`, `\n`, `\r` funktionieren.  
    Für **literales** `\t`: `\\t`
  - **Optionale Gruppen:** Sind sie **nicht gematcht**, setzt `patwatch` an der Stelle **leer** ein.  
    Optionale Deko wie z. B. Klammern/Komma **nur dann**, wenn sie **in die optionalen Gruppen** sind
  - **Wiederholte Gruppen** (z. B. `(…)+`) liefern in Python nur den **letzten** Treffer der Gruppe.  
    Wenn mehrere entfernte Werte eingesammelt werden sollen, nutzt man **Lookaheads** wie in Beispiel 2/10.




Transform-Pipeline (Spalte 5):
- Beispiel:  upper() | replace(\s+,_ ,re) | slice(0,8)
- Pro ID individuell; wird auf das erzeugte „Zeilen-Wort“ angewandt.
- (Auszug) Funktionen: upper, lower, first(n), last(n), slice(a,b), strip/lstrip/rstrip,
  replace_str(old,new), replace(rx,repl,flag), collapse_ws, zfill, padleft/padright,
  split/rsplit(delim,idx), rextract(rx,group), ensure_prefix/suffix, tr(src,dst)

Live-Ansicht mit Taste 'a':
- Normalmodus: Head/Tail mit --between nur bei Lücke (wie bisher).
- Alternativmodus: transformierte Kurz-Wörter; Terminalbreite wird ausgenutzt.
  Inhalt links (früheste), rechts (letzte), in der **Mitte** steht immer --between.
"""

import sys, re, argparse, subprocess, time, shutil, os, select
from collections import deque
from typing import List, Optional, Pattern

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Zeilen zählen & Wörter sammeln nach Regex-Patterns (pro ID).")
    p.add_argument("-p","--patterns", required=True,
                   help="Pattern-Datei: ID<TAB>LINE_REGEX<TAB>WORD_REGEX(optional)<TAB>TEMPLATE(optional)<TAB>TRANSFORMS(optional)")
    p.add_argument("-m","--maxw", type=int, default=10, help="Max. #Head-Wörter (erste N Treffer)")
    p.add_argument("-l","--lastw", type=int, default=10, help="Max. #Tail-Wörter (letzte M Treffer)")
    p.add_argument("--sep", default=" ", help="Trennzeichen zwischen Wörtern (Default: ' ')")
    p.add_argument("--between", default=" ... ", help="Trenner zwischen Head/Tail bzw. Mitte in Alternativansicht (Default: ' ... ')")
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

def apply_template(tmpl: str, m: re.Match) -> str:
    """Backrefs in tmpl mittels Match m ersetzen. Unterstützt: \1..\99, \g<name>, \\ \t \n \r."""
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
            out.append(m.group(idx) or "" if idx is not None else "")
            i = j; continue
        if c == 'g' and i + 1 < L and s[i+1] == '<':
            k = s.find('>', i+2)
            if k != -1:
                name = s[i+2:k]
                try:
                    out.append(m.group(name) or "")
                except IndexError:
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
    if not s: return []
    tokens, cur, esc = [], [], False
    for ch in s:
        if esc: cur.append(ch); esc=False; continue
        if ch == '\\': esc=True; continue
        if ch == '|': tokens.append(''.join(cur).strip()); cur=[]; continue
        cur.append(ch)
    if cur: tokens.append(''.join(cur).strip())
    return [t for t in tokens if t]

def split_call(token: str):
    m = re.match(r'^([a-zA-Z_]\w*)\s*(?:\((.*)\))?$', token)
    if not m: return token, []
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

def apply_pipeline(word: str, pipeline: str) -> str:
    funcs = {
        "upper": lambda s: s.upper(),
        "lower": lambda s: s.lower(),
        "title": lambda s: s.title(),
        "swapcase": lambda s: s.swapcase(),
        "first": lambda s,n="1": s[:int(n)],
        "last":  lambda s,n="1": s[-int(n):] if s else s,
        "slice": lambda s,a="",b="": s[(int(a) if a!="" else None):(int(b) if b!="" else None)],
        "strip": lambda s,chs="": s.strip(chs) if chs else s.strip(),
        "lstrip": lambda s,chs="": s.lstrip(chs) if chs else s.lstrip(),
        "rstrip": lambda s,chs="": s.rstrip(chs) if chs else s.rstrip(),
        "collapse_ws": lambda s: " ".join(s.split()),
        "replace_str": lambda s,a,b: s.replace(a,b),
        "replace": lambda s,rx,repl,flag="": re.sub(rx, repl, s, flags=(re.I if ('i' in flag.lower()) else 0)),
        "tr": lambda s,src,dst: s.translate(str.maketrans(src, dst)),
        "split": lambda s,delim,idx="0": (s.split(delim)[int(idx)] if delim in s and len(s.split(delim))>int(idx) else ""),
        "rsplit": lambda s,delim,idx="0": (s.rsplit(delim)[int(idx)] if delim in s and len(s.rsplit(delim))>int(idx) else ""),
        "rextract": lambda s,rx,grp="1": (re.search(rx,s).group(int(grp)) if re.search(rx,s) else ""),
        "padleft": lambda s,w,ch=" ": s.rjust(int(w), ch[:1] if ch else " "),
        "padright": lambda s,w,ch=" ": s.ljust(int(w), ch[:1] if ch else " "),
        "zfill": lambda s,w: s.zfill(int(w)),
        "ensure_prefix": lambda s,p: s if s.startswith(p) else (p+s),
        "ensure_suffix": lambda s,p: s if s.endswith(p) else (s+p),
        # einfache Zahlen-ops (best effort)
        "int": lambda s: str(int(float(s))) if s.strip() else s,
        "float": lambda s: str(float(s)) if s.strip() else s,
        "round": lambda s,d="0": (("{0:." + str(int(d)) + "f}").format(round(float(s), int(d)))) if s.strip() else s,
    }
    try:
        for tok in parse_pipeline(pipeline):
            name, args = split_call(tok)
            f = funcs.get(name)
            if not f: continue
            word = f(word, *args)
    except Exception:
        pass
    return word

# ---------- Datenstruktur ----------
class PatternRec:
    __slots__ = ("pid","line_re","word_re","tmpl","transforms",
                 "count","head","head_count","tail","tail_max","alts")
    def __init__(self, pid: str, line_re: Pattern, word_re: Optional[Pattern],
                 tmpl: str, transforms: str, maxw: int, lastw: int):
        self.pid = pid
        self.line_re = line_re
        self.word_re = word_re
        self.tmpl = tmpl
        self.transforms = transforms  # 5. Spalte
        self.count = 0
        self.head: List[str] = []
        self.head_count = 0
        self.tail_max = max(0, lastw)
        self.tail = deque(maxlen=lastw) if lastw > 0 else deque()
        self.alts: List[str] = []  # komplette Historie alternativer Wörter (für Alt-Ansicht)

def compile_rx(rx: str, flags: int) -> Pattern:
    return re.compile(rx, flags)

def pick_word(line: str, pat: PatternRec, cg_sep: str) -> str:
    if pat.word_re is not None:
        m = pat.word_re.search(line)
        if not m: return ""
        if pat.tmpl:
            return apply_template(pat.tmpl, m)
        if m.lastindex:
            parts = [g for g in m.groups() if g]
            if parts: return cg_sep.join(parts)
            return m.group(0)
        return m.group(0)
    return last_word(line)

def load_patterns(path: str, fs: str, flags: int, maxw: int, lastw: int) -> List[PatternRec]:
    pats: List[PatternRec] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, raw in enumerate(f, 1):
            line = raw.rstrip("\n")
            if not line.strip(): continue
            if line.endswith("\r"): line = line[:-1]
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
        pat.head_count = 0
        pat.alts.clear()
        if pat.tail_max > 0: pat.tail.clear()

    for line in input_text.splitlines():
        for pat in patterns:
            if pat.line_re.search(line):
                pat.count += 1
                w = pick_word(line, pat, cg_sep)
                if strip_p and w: w = strip_punct(w)
                # Transformiertes Alternativwort
                alt = apply_pipeline(w, pat.transforms) if pat.transforms else w
                pat.alts.append(alt)
                # Normal-Head/Tail
                if w:
                    if pat.head_count < maxw:
                        pat.head.append(w); pat.head_count += 1
                    if pat.tail_max > 0:
                        pat.tail.append(w)

    # Normalansicht-Rendering (Head/Tail mit Lücken-Logik)
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
                if skipped < overlap: skipped += 1; continue
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

# ---------- Alternativ-Rendering (Terminalbreite, Mitte=between) ----------
def render_alt_view(patterns: List[PatternRec], sep: str, between: str) -> str:
    cols = shutil.get_terminal_size(fallback=(80, 24)).columns
    lines = []
    for pat in patterns:
        prefix = f"{pat.pid}\t{pat.count}\t"
        # Verfügbarer Platz nach dem Prefix:
        avail = max(0, cols - len(prefix))
        if avail <= len(between):
            # zu eng, nur Prefix + evtl. abgeschnittenes between
            content = between[:avail]
            lines.append(prefix + content + "\n")
            continue

        left_cap = (avail - len(between)) // 2
        right_cap = avail - len(between) - left_cap

        # Links: früheste alt-Wörter
        left_words, cur = [], 0
        for w in pat.alts:
            add = len(w) if not left_words else len(sep) + len(w)
            if cur + add > left_cap: break
            left_words.append(w); cur += add
        left_count = len(left_words)

        # Rechts: letzte alt-Wörter (ohne Überschneidung mit links)
        right_words_rev, cur = [], 0
        i = len(pat.alts) - 1
        while i >= left_count:
            w = pat.alts[i]
            add = len(w) if not right_words_rev else len(sep) + len(w)
            if cur + add > right_cap: break
            right_words_rev.append(w); cur += add
            i -= 1
        right_words = list(reversed(right_words_rev))

        content = sep.join(left_words) + between + sep.join(right_words)
        lines.append(prefix + content + ("\n" if not content.endswith("\n") else ""))
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
        # 1) Daten holen (erst rechnen, NICHT sofort clearn → kein Flackern)
        text = run_cmd(args.cmd, args.shell, args.timeout, args.no_warn) if args.cmd else sys.stdin.read()
        normal_out = process_text(text, patterns, args.maxw, sep, between, args.strip_punct, cg_sep)
        # 2) Rendern je Modus
        content = render_alt_view(patterns, sep, between) if alt_mode else normal_out

        # 3) Aux-Block
        frame_body = content
        if args.auxcmd:
            aux_to = args.aux_timeout if args.aux_timeout is not None else args.timeout
            aux_out = run_cmd(args.auxcmd, args.shell, aux_to, args.no_warn)
            sep_line = aux_sep + ("" if aux_sep.endswith("\n") else "\n")
            aux_block = sep_line + (aux_out if aux_out.endswith("\n") else aux_out + "\n")
            frame_body = (aux_block + content) if args.aux_before else (content + aux_block)

        # 4) Header bauen (falls --clear)
        frame = frame_body
        if args.clear:
            left = (f"Every {args.interval:.1f}s: {args.cmd}" if args.cmd else "STDIN")
            mode_tag = " ALT" if alt_mode else ""
            right = "  ".join([p for p in (args.header + mode_tag if args.header or mode_tag else "", now_str(args.utc)) if p])
            header = build_header_line(left, right, color=args.color_header) + "\n"
            frame = "\x1b[H\x1b[2J" + header + frame_body

        sys.stdout.write(frame)
        sys.stdout.flush()

    # Watch-Loop (+ Keypoll für 'a')
    if args.interval and args.interval > 0:
        if not args.cmd:
            sys.stderr.write("[ERROR] --interval erfordert --cmd (STDIN kann nicht periodisch gelesen werden).\n")
            sys.exit(2)
        try:
            with KeyPoller() as kp:
                while True:
                    # Key-Poll (nonblocking)
                    keys = kp.poll()
                    if keys:
                        if 'a' in keys: alt_mode = not alt_mode
                    one_frame()
                    time.sleep(args.interval)
        except KeyboardInterrupt:
            pass
    else:
        # Einmalig (kein Toggle sinnvoll, da kein Refresh)
        one_frame()

if __name__ == "__main__":
    main()
