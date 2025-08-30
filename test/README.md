# Test-System für patwatch-alternative.py

Dieses Verzeichnis enthält alle Test-Daten und -Patterns für das patwatch-Programm.

## Dateien

### Test-Daten
- `test_log.txt` - Beispiel-Log-Datei mit verschiedenen Log-Levels und Services

### Pattern-Dateien
- `patterns_simple.txt` - Einfache Log-Level Patterns
- `patterns_working_words.txt` - Wort-Extraktion mit Transformationen
- `patterns_advanced_simple.txt` - Erweiterte Patterns ohne Regex-Fehler
- `patterns_complex_working.txt` - Komplexe Transform-Pipelines
- `patterns_complex_color_fixed.txt` - Komplexe Patterns für Farb-Tests
- `patterns_error_test.txt` - Patterns mit absichtlichen Fehlern (für Fehlertests)

### Tools
- `dynamic_log_generator.py` - Dynamischer Log-Generator für Live-Tests

## Verwendung

Alle Tests können über das Makefile im Hauptverzeichnis ausgeführt werden:

```bash
# Alle Tests ausführen
make test-all

# Einzelne Test-Kategorien
make test-basic      # Grundlegende Funktionalität
make test-words      # Wort-Extraktion
make test-advanced   # Erweiterte Patterns
make test-errors     # Fehlerbehandlung
make test-complex    # Komplexe Transformationen
make test-params     # Parameter-Validierung
make test-performance # Performance-Tests
make test-generator  # Log-Generator testen

# Interaktive Tests
make test-watch      # Watch-Modus testen
make repeatcolor     # Komplexer Farb-Test mit dynamischen Logs

# Hilfe anzeigen
make help
```

## Spezielle Tests

### repeatcolor - Komplexer Farb-Test
```bash
make repeatcolor
```

Dieser Test startet patwatch mit:
- **Dynamischen Logs**: Kontinuierlich generierte, komplexe Log-Zeilen
- **Farbige Ausgabe**: ANSI-256 Farbchips für alle Wörter
- **Komplexe Patterns**: Verschiedene Transform-Pipelines
- **Watch-Modus**: 1-Sekunden-Intervall mit Bildschirm löschen
- **Interaktive Steuerung**: Taste 'a' für Alternativansicht

**Features:**
- 10 verschiedene Services mit unterschiedlichen Aktionen
- Realistische Log-Daten (Emails, IPs, Beträge, IDs)
- Verschiedene Status (successful, failed, timeout)
- Komplexe Transform-Pipelines (upper, lower, padleft, etc.)

**Steuerung:**
- `a` - Wechsel zwischen Normal- und Alternativansicht
- `Ctrl+C` - Beenden

## Pattern-Datei Format

Pattern-Dateien verwenden das Tab-getrennte Format:
```
ID<TAB>LINE_REGEX<TAB>WORD_REGEX(optional)<TAB>TEMPLATE(optional)<TAB>TRANSFORMS(optional)
```

### Beispiele:
```
ERROR	ERROR	\w+		upper
INFO	INFO	\w+		lower
UserService	\[UserService\]	\w+		upper|first(3)
```

## Bekannte Probleme

- `\1` in der WORD_REGEX-Spalte verursacht Fehler (sollte in TEMPLATE-Spalte sein)
- Komplexe Transform-Pipelines benötigen korrekte Parameter
- Capture-Groups in WORD_REGEX können zu Problemen führen

## Fehlerbehandlung

Das Programm zeigt Warnungen für:
- Ungültige Regex-Patterns
- Fehlende Transform-Parameter
- Nicht existierende Dateien
- Ungültige Kommandos

Diese Warnungen sind normal und zeigen, dass die Exception-Behandlung funktioniert.
