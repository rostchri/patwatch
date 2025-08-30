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
    Wenn mehrere entfernte Werte eingesammelt werden sollen, nutzt man **Lookaheads**
    
Transforms (Spalte 5):
- Pipeline via |, z. B.:  upper() | replace(\s+,_ ,i) | slice(0,8)
- Der erste Token darf ein Backref-/Template-String sein (z. B. "\2", "[\1]-\3").
  Dann startet die Pipeline mit diesem String statt dem zuvor gebildeten Wort.
- Zusätzlich: set()/append()/prepend()/concat() akzeptieren Template-Argumente.

Wortbildung:
- Ohne WORD_REGEX → letztes Wort der Zeile.
- Mit WORD_REGEX:
    * Mit TEMPLATE (Spalte 4): Wort via Backrefs (\1..\99, \g<name>), Escapes \\ \t \n \r.
    * Ohne TEMPLATE: alle Capture-Gruppen (1..n) werden mit --cg-sep zusammengefügt (leere Gruppen übersprungen);
      gibt es keine Gruppen → gesamter Match.

Live-Ansicht (Taste 'a'):
- Normalmodus: Head/Tail mit Lückenlogik und --between nur bei echter Lücke.
- Alternativmodus: transformierte Kurz-Wörter; Terminalbreite wird maximal genutzt.
  Links die frühesten, rechts die letzten Alt-Wörter; in der Mitte steht exakt --between.
"""

import sys, re, argparse, subprocess, time, shutil
from collections import deque
from typing import List, Optional, Pattern

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Zeilen zählen & Wörter sammeln nach Regex-Patterns (pro ID).")
    p.add_argument("-p","--patterns", required=True, help="Pattern-Datei: ID<TAB>LINE_REGEX<TAB>WORD_REGEX(optional)<TAB>TEMPLATE(optional)")
    p.add_argument("-m","--maxw", type=int, default=10, help="Max. #Head-Wörter (erste N Treffer)")
    p.add_argument("-l","--lastw", type=int, default=10, help="Max. #Tail-Wörter (letzte M Treffer)")
    p.add_argument("--sep", default=" ", help="Trennzeichen zwischen Wörtern (Default: ' ')")
    p.add_argument("--between", default=" ... ", help="Trenner zwischen Head und Tail (Default: ' ... ')")
    p.add_argument("-i","--ignorecase", action="store_true", help="Case-insensitive Matching")
    p.add_argument("--strip-punct", action="store_true", help="Satzzeichen am Wortanfang/-ende entfernen")
    p.add_argument("--fs", default="\\t", help="Feldtrenner in Pattern-Datei (Default: '\\t')")
    p.add_argument("--cg-sep", default="", help="Trenner beim Zusammenfügen MEHRERER Capture-Gruppen (wenn kein TEMPLATE)")

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

    # Zusatzkommando
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
    """Ersetzt Backrefs in tmpl mittels Match m.
       Unterstützt: \1..\9..\10, \g<name>, \\, \t, \n, \r.
    """
    out = []
    i = 0
    s = tmpl
    L = len(s)
    while i < L:
        ch = s[i]
        if ch != '\\':
            out.append(ch); i += 1; continue
        i += 1
        if i >= L:
            out.append('\\'); break
        c = s[i]

        # Numerische Backrefs \1 .. \99...
        if c.isdigit():
            j = i
            while j < L and s[j].isdigit():
                j += 1
            idx = int(s[i:j]) if j > i else None
            out.append(m.group(idx) or "" if idx is not None else "")
            i = j
            continue

        # \g<name>
        if c == 'g' and i + 1 < L and s[i+1] == '<':
            j = i + 2
            k = s.find('>', j)
            if k != -1:
                name = s[j:k]
                out.append(m.groupdict().get(name, "") or m.group(name) if name in m.re.groupindex else "")
                i = k + 1
                continue
            # kein '>' gefunden → literal
            out.append('\\g'); i += 1; continue

        # Standard-Escapes
        if c == 't': out.append('\t'); i += 1; continue
        if c == 'n': out.append('\n'); i += 1; continue
        if c == 'r': out.append('\r'); i += 1; continue
        if c == '\\': out.append('\\'); i += 1; continue

        # Unbekanntes Escape -> nur Zeichen selbst
        out.append(c); i += 1

    return "".join(out)

class PatternRec:
    __slots__ = ("pid","line_re","word_re","tmpl","count","head","head_count","tail","tail_max")
    def __init__(self, pid: str, line_re: Pattern, word_re: Optional[Pattern], tmpl: str,
                 maxw: int, lastw: int):
        self.pid = pid
        self.line_re = line_re
        self.word_re = word_re
        self.tmpl = tmpl  # 4. Spalte (kann "")
        self.count = 0
        self.head: List[str] = []
        self.head_count = 0
        self.tail_max = max(0, lastw)
        self.tail = deque(maxlen=lastw) if lastw > 0 else deque()

def compile_rx(rx: str, flags: int) -> Pattern:
    return re.compile(rx, flags)

def pick_word(line: str, pat: PatternRec, cg_sep: str) -> str:
    """Erzeugt das Wort gemäß WORD_REGEX + optionaler TEMPLATE."""
    if pat.word_re is not None:
        m = pat.word_re.search(line)
        if not m:
            return ""
        # TEMPLATE gewählt → Backrefs anwenden
        if pat.tmpl:
            return apply_template(pat.tmpl, m)

        # kein TEMPLATE → Standardlogik
        if m.lastindex:  # es gibt Gruppen
            parts = [g for g in m.groups() if g]  # leere Gruppen überspringen
            if parts:
                return cg_sep.join(parts)
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
            # bis zu 4 Spalten zulassen
            parts = line.split(fs, 3)
            if len(parts) < 2:
                sys.stderr.write(f"[WARN] Zeile {ln}: erwarte mind. 2 Felder (ID{fs}LINE_REGEX[ {fs}WORD_REGEX[ {fs}TEMPLATE ]]). Übersprungen.\n")
                continue
            pid = parts[0].strip()
            line_rx = posix_to_py(parts[1])
            word_rx = posix_to_py(parts[2]) if len(parts)>=3 and parts[2] != "" else None
            tmpl = parts[3] if len(parts) >= 4 else ""
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
            pats.append(PatternRec(pid, lre, wre, tmpl, maxw, lastw))
    return pats

# ---------- Kernverarbeitung ----------
def process_text(input_text: str, patterns: List[PatternRec], maxw: int, sep: str, between: str,
                 strip_p: bool, cg_sep: str) -> str:
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
                w = pick_word(line, pat, cg_sep)
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

    def one_run():
        # 1) Daten holen & verarbeiten (erst rechnen, NICHT sofort clearn → kein Flackern)
        text = run_cmd(args.cmd, args.shell, args.timeout, args.no_warn) if args.cmd else sys.stdin.read()
        main_out = process_text(text, patterns, args.maxw, sep, between, args.strip_punct, cg_sep)

        # 2) Aux-Output ggf. anhängen (Position via --aux-before)
        frame_body = main_out
        if args.auxcmd:
            aux_to = args.aux_timeout if args.aux_timeout is not None else args.timeout
            aux_out = run_cmd(args.auxcmd, args.shell, aux_to, args.no_warn)
            sep_line = aux_sep + ("" if aux_sep.endswith("\n") else "\n")
            aux_block = sep_line + (aux_out if aux_out.endswith("\n") else aux_out + "\n")
            frame_body = (aux_block + main_out) if args.aux_before else (main_out + aux_block)

        # 3) Header-String bauen (falls --clear)
        frame = frame_body
        if args.clear:
            left = (f"Every {args.interval:.1f}s: {args.cmd}" if args.cmd else "STDIN")
            right = "  ".join([p for p in (args.header, now_str(args.utc)) if p])
            header = build_header_line(left, right, color=args.color_header) + "\n"
            # 4) Atomar ausgeben: Clear + Header + Frame in EINEM write → kein Flackern
            frame = "\x1b[H\x1b[2J" + header + frame_body

        sys.stdout.write(frame)
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
