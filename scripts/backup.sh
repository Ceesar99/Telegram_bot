#!/bin/bash
# Automated Backup Script for Trading Bot
# Creates timestamped backups of critical system data

set -e  # Exit on any error

# Configuration
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/workspace/backup"
LOG_FILE="/workspace/logs/backup.log"
RETENTION_DAYS=30

# Functions
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

create_backup_dir() {
    local backup_path="$BACKUP_DIR/$DATE"
    mkdir -p "$backup_path"
    echo "$backup_path"
}

backup_databases() {
    local backup_path="$1"
    log "Backing up databases..."
    
    if [ -d "/workspace/data" ]; then
        cp /workspace/data/*.db "$backup_path/" 2>/dev/null || log "No database files found"
        log "Database backup completed"
    else
        log "Warning: Data directory not found"
    fi
}

backup_logs() {
    local backup_path="$1"
    log "Backing up recent logs..."
    
    # Backup logs from last 7 days
    find /workspace/logs -name "*.log" -mtime -7 -exec cp {} "$backup_path/" \; 2>/dev/null || true
    find /workspace/logs -name "*.json" -mtime -7 -exec cp {} "$backup_path/" \; 2>/dev/null || true
    
    log "Log backup completed"
}

backup_models() {
    local backup_path="$1"
    log "Backing up AI models..."
    
    if [ -d "/workspace/models" ] && [ "$(ls -A /workspace/models)" ]; then
        mkdir -p "$backup_path/models"
        cp -r /workspace/models/* "$backup_path/models/" 2>/dev/null || true
        log "Model backup completed"
    else
        log "No models to backup"
    fi
}

backup_config() {
    local backup_path="$1"
    log "Backing up configuration (sanitized)..."
    
    # Backup configuration without sensitive data
    python3 -c "
import sys
sys.path.append('/workspace')
try:
    from config_manager import config_manager
    import json
    config = config_manager.export_config(include_sensitive=False)
    with open('$backup_path/config_backup.json', 'w') as f:
        json.dump(config, f, indent=2)
    print('Configuration backup created')
except Exception as e:
    print(f'Config backup failed: {e}')
" 2>/dev/null || log "Config backup failed"
}

cleanup_old_backups() {
    log "Cleaning up old backups (older than $RETENTION_DAYS days)..."
    find "$BACKUP_DIR" -type d -name "202*" -mtime +$RETENTION_DAYS -exec rm -rf {} \; 2>/dev/null || true
    log "Cleanup completed"
}

create_backup_info() {
    local backup_path="$1"
    cat > "$backup_path/backup_info.txt" << EOF
Backup Information
==================
Date: $(date)
Backup Path: $backup_path
System: $(uname -a)
Python Version: $(python3 --version)
Disk Usage: $(df -h /workspace | tail -1)

Contents:
$(ls -la "$backup_path")
EOF
}

compress_backup() {
    local backup_path="$1"
    log "Compressing backup..."
    
    cd "$BACKUP_DIR"
    tar -czf "${DATE}.tar.gz" "$DATE/" 2>/dev/null || {
        log "Warning: Compression failed, keeping uncompressed backup"
        return 1
    }
    
    # Remove uncompressed directory if compression succeeded
    rm -rf "$DATE/"
    log "Backup compressed to ${DATE}.tar.gz"
}

# Main execution
main() {
    log "Starting backup process..."
    
    # Create backup directory
    BACKUP_PATH=$(create_backup_dir)
    log "Backup directory: $BACKUP_PATH"
    
    # Perform backups
    backup_databases "$BACKUP_PATH"
    backup_logs "$BACKUP_PATH"
    backup_models "$BACKUP_PATH"
    backup_config "$BACKUP_PATH"
    
    # Create backup information
    create_backup_info "$BACKUP_PATH"
    
    # Compress backup
    compress_backup "$BACKUP_PATH"
    
    # Cleanup old backups
    cleanup_old_backups
    
    log "Backup process completed successfully"
    log "Backup location: $BACKUP_DIR/${DATE}.tar.gz"
    
    # Display backup size
    if [ -f "$BACKUP_DIR/${DATE}.tar.gz" ]; then
        SIZE=$(du -h "$BACKUP_DIR/${DATE}.tar.gz" | cut -f1)
        log "Backup size: $SIZE"
    fi
}

# Error handling
trap 'log "Backup failed with error on line $LINENO"' ERR

# Run main function
main "$@"

# Success exit
log "Backup script completed"
exit 0