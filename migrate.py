#!/usr/bin/env python3
"""Run once before updating the web and worker services in production."""
from database import apply_schema_migrations

if __name__ == "__main__":
    apply_schema_migrations()
    print("Database schema migration completed.")