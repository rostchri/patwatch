# Makefile für patwatch Tests
# Verwendung: make <target>

# Variablen
PYTHON = python3
SCRIPT = patwatch-alternative.py
TEST_DIR = test
LOG_FILE = $(TEST_DIR)/test_log.txt

# Standard-Target
.PHONY: all
all: test-basic test-words test-advanced test-errors test-complex

# Grundlegende Tests
.PHONY: test-basic
test-basic:
	@echo "=== Grundlegende Tests ==="
	@echo "Test: Einfache Pattern-Verarbeitung"
	$(PYTHON) $(SCRIPT) -p $(TEST_DIR)/patterns_simple.txt -c "cat $(LOG_FILE)"
	@echo ""

# Wort-Extraktion Tests
.PHONY: test-words
test-words:
	@echo "=== Wort-Extraktion Tests ==="
	@echo "Test: Wort-Extraktion mit Transformationen"
	$(PYTHON) $(SCRIPT) -p $(TEST_DIR)/patterns_working_words.txt -c "cat $(LOG_FILE)" --maxw 3 --lastw 2
	@echo ""
	@echo "Test: Wort-Extraktion mit Farben"
	$(PYTHON) $(SCRIPT) -p $(TEST_DIR)/patterns_working_words.txt -c "cat $(LOG_FILE)" --maxw 3 --lastw 2 --color
	@echo ""

# Erweiterte Tests
.PHONY: test-advanced
test-advanced:
	@echo "=== Erweiterte Tests ==="
	@echo "Test: Komplexe Pattern-Verarbeitung"
	$(PYTHON) $(SCRIPT) -p $(TEST_DIR)/patterns_advanced_simple.txt -c "cat $(LOG_FILE)" --no-warn
	@echo ""

# Fehler-Tests
.PHONY: test-errors
test-errors:
	@echo "=== Fehler-Tests ==="
	@echo "Test: Nicht existierende Pattern-Datei"
	$(PYTHON) $(SCRIPT) -p $(TEST_DIR)/nonexistent.txt -c "cat $(LOG_FILE)" || echo "Erwarteter Fehler: Datei nicht gefunden"
	@echo ""
	@echo "Test: Ungültige Kommandos"
	$(PYTHON) $(SCRIPT) -p $(TEST_DIR)/patterns_simple.txt -c "nonexistent_command" --no-warn
	@echo ""
	@echo "Test: Pattern-Datei mit Fehlern"
	$(PYTHON) $(SCRIPT) -p $(TEST_DIR)/patterns_error_test.txt -c "cat $(LOG_FILE)" --no-warn
	@echo ""

# Komplexe Transform-Tests
.PHONY: test-complex
test-complex:
	@echo "=== Komplexe Transform-Tests ==="
	@echo "Test: Erweiterte Transform-Pipelines"
	$(PYTHON) $(SCRIPT) -p $(TEST_DIR)/patterns_complex_working.txt -c "cat $(LOG_FILE)" --maxw 5 --lastw 3
	@echo ""

# Parameter-Tests
.PHONY: test-params
test-params:
	@echo "=== Parameter-Tests ==="
	@echo "Test: Negative Parameter (sollte Fehler geben)"
	$(PYTHON) $(SCRIPT) -p $(TEST_DIR)/patterns_simple.txt -c "cat $(LOG_FILE)" --maxw -1 || echo "Erwarteter Fehler: Negative Parameter"
	@echo ""
	@echo "Test: Verschiedene Separator"
	$(PYTHON) $(SCRIPT) -p $(TEST_DIR)/patterns_working_words.txt -c "cat $(LOG_FILE)" --sep " | " --maxw 2
	@echo ""

# Watch-Modus Tests
.PHONY: test-watch
test-watch:
	@echo "=== Watch-Modus Tests ==="
	@echo "Test: Watch-Modus (2 Sekunden, 3 Durchläufe)"
	@echo "Drücke Ctrl+C nach 3 Durchläufen..."
	$(PYTHON) $(SCRIPT) -p $(TEST_DIR)/patterns_working_words.txt -c "cat $(LOG_FILE)" -t 2 --clear --maxw 2 || echo "Watch-Modus beendet"

# Komplexer Farb-Test mit dynamischen Logs
.PHONY: repeatcolor
repeatcolor:
	@echo "=== Komplexer Farb-Test mit dynamischen Logs ==="
	@echo "Starte patwatch mit komplexen Patterns und dynamischen Logs..."
	@echo "Features:"
	@echo "  - Farbige Wort-Chips (--color)"
	@echo "  - Dynamische Log-Generierung"
	@echo "  - Komplexe Transform-Pipelines"
	@echo "  - Watch-Modus mit 1s Intervall"
	@echo "  - Bildschirm löschen (--clear)"
	@echo "  - Head/Tail Anzeige (--maxw 5 --lastw 3)"
	@echo ""
	@echo "Tastatur-Steuerung:"
	@echo "  - Taste 'a': Wechsel zwischen Normal- und Alternativansicht"
	@echo "  - Ctrl+C: Beenden"
	@echo ""
	@echo "Starte in 3 Sekunden..."
	@sleep 3
	$(PYTHON) $(SCRIPT) -p $(TEST_DIR)/patterns_complex_color_fixed.txt -c "$(PYTHON) $(TEST_DIR)/dynamic_log_generator.py | head -20" -t 1 --clear --color --maxw 5 --lastw 3 --color-header

# Pipe-Test mit Timeout-Generator
.PHONY: piperepeatcolor
piperepeatcolor:
	@echo "=== Pipe-Test mit Timeout-Generator ==="
	@echo "Starte patwatch mit Pipe-Eingabe und Timeout-Generator..."
	@echo "Features:"
	@echo "  - Pipe-Eingabe von dynamischem Log-Generator"
	@echo "  - Timeout nach 10 Sekunden"
	@echo "  - Farbige Wort-Chips (--color)"
	@echo "  - Komplexe Transform-Pipelines"
	@echo "  - Einmalige Verarbeitung (kein Watch-Modus)"
	@echo "  - Bildschirm löschen (--clear)"
	@echo "  - Head/Tail Anzeige (--maxw 5 --lastw 3)"
	@echo ""
	@echo "Tastatur-Steuerung:"
	@echo "  - Taste 'a': Wechsel zwischen Normal- und Alternativansicht"
	@echo "  - Ctrl+C: Beenden"
	@echo ""
	@echo "Starte in 3 Sekunden..."
	@sleep 3
	$(PYTHON) $(TEST_DIR)/dynamic_log_generator.py --timeout 10 | $(PYTHON) $(SCRIPT) -p $(TEST_DIR)/patterns_complex_color_fixed.txt --clear --color --maxw 5 --lastw 3 --color-header

# CMD-Test mit Timeout-Generator
.PHONY: cmdrepeatcolor
cmdrepeatcolor:
	@echo "=== CMD-Test mit Timeout-Generator ==="
	@echo "Starte patwatch mit CMD-Eingabe und Timeout-Generator..."
	@echo "Features:"
	@echo "  - Kommando-Eingabe von dynamischem Log-Generator"
	@echo "  - Kontinuierliche Daten (5 Zeilen pro Durchlauf)"
	@echo "  - Farbige Wort-Chips (--color)"
	@echo "  - Komplexe Transform-Pipelines"
	@echo "  - Watch-Modus mit 1s Intervall"
	@echo "  - Bildschirm löschen (--clear)"
	@echo "  - Head/Tail Anzeige (--maxw 5 --lastw 3)"
	@echo ""
	@echo "Tastatur-Steuerung:"
	@echo "  - Taste 'a': Wechsel zwischen Normal- und Alternativansicht"
	@echo "  - Ctrl+C: Beenden"
	@echo ""
	@echo "Starte in 3 Sekunden..."
	@sleep 3
	$(PYTHON) $(SCRIPT) -p $(TEST_DIR)/patterns_complex_color_fixed.txt -c "$(PYTHON) $(TEST_DIR)/dynamic_log_generator.py --timeout 0 | head -5" -t 1 --clear --color --maxw 5 --lastw 3 --color-header

# Pipe-Test mit unendlichem Generator
.PHONY: piperepeatcolorinfinite
piperepeatcolorinfinite:
	@echo "=== Pipe-Test mit unendlichem Generator ==="
	@echo "Starte patwatch mit Pipe-Eingabe und unendlichem Log-Generator..."
	@echo "Features:"
	@echo "  - Kommando-Eingabe von dynamischem Log-Generator"
	@echo "  - Unendlicher Lauf (Timeout: 0 = nie terminieren)"
	@echo "  - Farbige Wort-Chips (--color)"
	@echo "  - Komplexe Transform-Pipelines"
	@echo "  - Watch-Modus mit 1s Intervall"
	@echo "  - Bildschirm löschen (--clear)"
	@echo "  - Head/Tail Anzeige (--maxw 5 --lastw 3)"
	@echo ""
	@echo "Tastatur-Steuerung:"
	@echo "  - Taste 'a': Wechsel zwischen Normal- und Alternativansicht"
	@echo "  - Ctrl+C: Beenden"
	@echo ""
	@echo "Starte in 3 Sekunden..."
	@sleep 3
	$(PYTHON) $(SCRIPT) -p $(TEST_DIR)/patterns_complex_color_fixed.txt -c "$(PYTHON) $(TEST_DIR)/dynamic_log_generator.py --timeout 0" -t 1 --clear --color --maxw 5 --lastw 3 --color-header

# Schneller Test des Log-Generators
.PHONY: test-generator
test-generator:
	@echo "=== Test des Log-Generators ==="
	@echo "Generiere 5 Log-Zeilen..."
	$(PYTHON) $(TEST_DIR)/dynamic_log_generator.py | head -5
	@echo ""

# Performance-Tests
.PHONY: test-performance
test-performance:
	@echo "=== Performance-Tests ==="
	@echo "Test: Große Log-Datei generieren..."
	@for i in {1..1000}; do echo "2024-01-15 10:30:$$i INFO [TestService] Test message $$i"; done > $(TEST_DIR)/large_log.txt
	@echo "Test: Verarbeitung großer Datei"
	$(PYTHON) $(SCRIPT) -p $(TEST_DIR)/patterns_simple.txt -c "cat $(TEST_DIR)/large_log.txt" --maxw 10 --lastw 10
	@rm -f $(TEST_DIR)/large_log.txt
	@echo ""

# Alle Tests ausführen
.PHONY: test-all
test-all: test-basic test-words test-advanced test-errors test-complex test-params test-performance repeatcolor piperepeatcolor cmdrepeatcolor
	@echo "=== Alle Tests abgeschlossen ==="

# Cleanup
.PHONY: clean
clean:
	@echo "Bereinige Test-Dateien..."
	@rm -f $(TEST_DIR)/large_log.txt
	@echo "Cleanup abgeschlossen"

# Hilfe
.PHONY: help
help:
	@echo "Verfügbare Targets:"
	@echo "  test-basic      - Grundlegende Funktionalität testen"
	@echo "  test-words      - Wort-Extraktion und Transformationen testen"
	@echo "  test-advanced   - Erweiterte Pattern-Verarbeitung testen"
	@echo "  test-errors     - Fehlerbehandlung testen"
	@echo "  test-complex    - Komplexe Transform-Pipelines testen"
	@echo "  test-params     - Parameter-Validierung testen"
	@echo "  test-watch      - Watch-Modus testen (interaktiv)"
	@echo "  test-generator  - Log-Generator testen"
	@echo "  repeatcolor     - Komplexer Farb-Test mit dynamischen Logs (interaktiv)"
	@echo "  piperepeatcolor - Pipe-Test mit Timeout-Generator (interaktiv)"
	@echo "  piperepeatcolorinfinite - Pipe-Test mit unendlichem Generator (interaktiv)"
	@echo "  cmdrepeatcolor  - CMD-Test mit Timeout-Generator (interaktiv)"
	@echo "  test-performance- Performance mit großen Dateien testen"
	@echo "  test-all        - Alle Tests ausführen"
	@echo "  clean           - Test-Dateien bereinigen"
	@echo "  help            - Diese Hilfe anzeigen"
