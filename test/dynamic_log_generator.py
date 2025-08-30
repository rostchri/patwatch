#!/usr/bin/env python3
"""
Dynamischer Log-Generator für patwatch Tests
Generiert kontinuierlich komplexe Log-Zeilen mit verschiedenen Services, Status und Daten.
"""

import time
import random
import sys
import argparse
from datetime import datetime, timedelta

# Services und deren mögliche Aktionen
SERVICES = {
    "UserService": ["login", "logout", "profile_update", "password_reset", "account_creation"],
    "DatabaseService": ["connection", "query_execution", "backup", "restore", "optimization"],
    "PaymentService": ["transaction", "refund", "subscription", "billing", "invoice"],
    "AuthService": ["authentication", "authorization", "token_refresh", "session_management"],
    "NotificationService": ["email_send", "push_notification", "sms_send", "webhook_call"],
    "CacheService": ["cache_hit", "cache_miss", "cache_eviction", "cache_warmup"],
    "FileService": ["file_upload", "file_download", "file_delete", "file_backup"],
    "APIService": ["rate_limit", "endpoint_call", "webhook_receive", "api_gateway"],
    "BackupService": ["backup_start", "backup_complete", "backup_failed", "restore_start"],
    "ReportService": ["report_generation", "report_export", "report_scheduling"]
}

# Status und deren Wahrscheinlichkeiten
STATUSES = {
    "successful": 0.7,
    "failed": 0.2,
    "timeout": 0.1
}

# Email-Domains
EMAIL_DOMAINS = ["example.com", "test.org", "demo.net", "sample.io", "user.dev"]

# IP-Ranges für verschiedene Services
IP_RANGES = {
    "UserService": ["192.168.1", "10.0.1", "172.16.1"],
    "DatabaseService": ["192.168.2", "10.0.2", "172.16.2"],
    "PaymentService": ["192.168.3", "10.0.3", "172.16.3"]
}

def generate_email():
    """Generiert eine zufällige Email-Adresse."""
    names = ["john", "jane", "admin", "user", "test", "demo", "sample", "guest"]
    name = random.choice(names)
    domain = random.choice(EMAIL_DOMAINS)
    return f"{name}@{domain}"

def generate_ip(service):
    """Generiert eine IP-Adresse basierend auf dem Service."""
    if service in IP_RANGES:
        base = random.choice(IP_RANGES[service])
    else:
        base = "192.168.0"
    return f"{base}.{random.randint(1, 254)}"

def generate_amount():
    """Generiert einen zufälligen Betrag."""
    return round(random.uniform(1.0, 9999.99), 2)

def generate_transaction_id():
    """Generiert eine zufällige Transaction-ID."""
    return f"TXN{random.randint(100000, 999999)}"

def generate_quantity():
    """Generiert eine zufällige Menge."""
    return random.randint(1, 1000)

def generate_log_line():
    """Generiert eine komplexe Log-Zeile."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    service = random.choice(list(SERVICES.keys()))
    action = random.choice(SERVICES[service])
    status = random.choices(list(STATUSES.keys()), weights=list(STATUSES.values()))[0]
    
    # Zufällig ein Betriebssystem hinzufügen
    os_name = random.choice(OPERATING_SYSTEMS) if random.random() < 0.3 else ""
    
    # Basis-Log-Zeile
    log_line = f"{timestamp} {status.upper()} [{service}] {action}"
    if os_name:
        log_line += f" on {os_name}"
    
    # Service-spezifische Details hinzufügen
    if service == "UserService":
        email = generate_email()
        log_line += f": {email}"
    elif service == "PaymentService":
        amount = generate_amount()
        transaction_id = generate_transaction_id()
        log_line += f": transaction_id={transaction_id}, amount={amount}"
    elif service == "DatabaseService":
        query_time = random.randint(10, 5000)
        log_line += f": query_time={query_time}ms"
    elif service == "NotificationService":
        email = generate_email()
        device_id = f"DEV{random.randint(1000, 9999)}"
        log_line += f": recipient={email}, device_id={device_id}"
    elif service == "AuthService":
        user_id = random.randint(1000, 9999)
        ip = generate_ip(service)
        log_line += f": user_id={user_id}, IP={ip}"
    elif service == "CacheService":
        cache_key = f"cache_key_{random.randint(1, 100)}"
        log_line += f": key={cache_key}"
    elif service == "FileService":
        file_size = random.randint(1024, 10485760)  # 1KB bis 10MB
        file_path = f"/var/files/file_{random.randint(1, 100)}.dat"
        log_line += f": path={file_path}, size={file_size}"
    elif service == "APIService":
        endpoint = f"/api/v{random.randint(1, 3)}/{random.choice(['users', 'data', 'config', 'status'])}"
        response_time = random.randint(50, 2000)
        log_line += f": endpoint={endpoint}, response_time={response_time}ms"
    elif service == "BackupService":
        backup_size = random.randint(1000000, 1000000000)  # 1MB bis 1GB
        log_line += f": backup_size={backup_size} bytes"
    elif service == "ReportService":
        report_id = f"REP-{datetime.now().year}-{random.randint(1, 999):03d}"
        log_line += f": report_id={report_id}"
    
    # Zusätzliche Details basierend auf Status
    if status == "failed":
        error_codes = ["E001", "E002", "E003", "E004", "E005"]
        error_msg = random.choice(["Connection timeout", "Invalid credentials", "Resource not found", "Permission denied", "Internal error"])
        log_line += f", error_code={random.choice(error_codes)}, error_msg=\"{error_msg}\""
    elif status == "timeout":
        timeout_duration = random.randint(30, 300)
        log_line += f", timeout_duration={timeout_duration}s"
    
    return log_line

def main():
    """Hauptfunktion - generiert kontinuierlich Log-Zeilen."""
    parser = argparse.ArgumentParser(description='Dynamischer Log-Generator für patwatch Tests')
    parser.add_argument('--timeout', '-t', type=float, default=0.0, 
                       help='Timeout in Sekunden (0 = nie terminieren, Standard: 0)')
    parser.add_argument('--interval', '-i', type=float, default=0.5,
                       help='Intervall zwischen Log-Zeilen in Sekunden (Standard: 0.5)')
    args = parser.parse_args()
    
    interval = args.interval
    timeout = args.timeout
    
    print(f"# Dynamischer Log-Generator gestartet - Drücke Ctrl+C zum Beenden", file=sys.stderr)
    print(f"# Intervall: {interval}s", file=sys.stderr)
    if timeout > 0:
        print(f"# Timeout: {timeout}s", file=sys.stderr)
    else:
        print(f"# Timeout: nie (läuft kontinuierlich)", file=sys.stderr)
    
    start_time = time.time()
    
    try:
        while True:
            # Timeout prüfen
            if timeout > 0 and (time.time() - start_time) >= timeout:
                print(f"# Timeout nach {timeout}s erreicht - Generator beendet", file=sys.stderr)
                break
                
            log_line = generate_log_line()
            print(log_line)
            try:
                sys.stdout.flush()
            except BrokenPipeError:
                # Normal wenn stdout geschlossen wird (z.B. durch head)
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        print(f"# Log-Generator beendet", file=sys.stderr)
        sys.exit(0)
    except BrokenPipeError:
        # Normal wenn stdout geschlossen wird
        sys.exit(0)

if __name__ == "__main__":
    main()
