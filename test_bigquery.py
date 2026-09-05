#!/usr/bin/env python3
"""
BigQuery Integration Test & Verification Script
Run this script to verify your Google Cloud BigQuery connection,
check table/dataset existence, insert a test login record, and query records back.

Usage:
    python backend/test_bigquery.py
or:
    ./venv/bin/python backend/test_bigquery.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

# Ensure we can import from services
backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))

# Load .env file
dotenv_path = backend_dir / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)
else:
    load_dotenv()

from services.bigquery_service import (
    get_bigquery_client,
    ensure_dataset_and_table,
    log_user_to_bigquery,
    get_bigquery_status,
    BIGQUERY_DATASET,
    BIGQUERY_TABLE,
    _resolve_credentials_path
)

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

def print_step(num: int, title: str):
    print(f"\n{BOLD}{CYAN}[Step {num}] {title}{RESET}")

def main():
    print(f"\n{BOLD}══════════════════════════════════════════════════════{RESET}")
    print(f"{BOLD}  FinAdvisor — BigQuery GCP Integration Diagnostics   {RESET}")
    print(f"{BOLD}══════════════════════════════════════════════════════{RESET}")

    # Step 1: Check Python Dependencies
    print_step(1, "Checking Python Dependencies")
    try:
        import google.cloud.bigquery
        print(f"  {GREEN}✔ google-cloud-bigquery is installed ({google.cloud.bigquery.__version__}){RESET}")
    except ImportError:
        print(f"  {RED}✘ google-cloud-bigquery is NOT installed.{RESET}")
        print(f"    Please run: pip install google-cloud-bigquery")
        sys.exit(1)

    # Step 2: Check Environment Configuration
    print_step(2, "Checking Environment Variables (.env)")
    project_id = os.getenv("GCP_PROJECT_ID")
    creds_env = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    dataset_name = os.getenv("BIGQUERY_DATASET", BIGQUERY_DATASET)
    table_name = os.getenv("BIGQUERY_TABLE", BIGQUERY_TABLE)

    print(f"  - GCP_PROJECT_ID: {project_id or f'{RED}Not set{RESET}'}")
    print(f"  - BIGQUERY_DATASET: {dataset_name}")
    print(f"  - BIGQUERY_TABLE: {table_name}")
    print(f"  - GOOGLE_APPLICATION_CREDENTIALS: {creds_env or f'{YELLOW}Not set (ADC will be used){RESET}'}")

    if not project_id or project_id == "YOUR_GCP_PROJECT_ID":
        print(f"\n  {YELLOW}⚠ Warning: GCP_PROJECT_ID is still set to placeholder 'YOUR_GCP_PROJECT_ID'.{RESET}")
        print(f"    Update GCP_PROJECT_ID in backend/.env with your actual GCP Project ID.")

    resolved_creds = _resolve_credentials_path()
    if creds_env:
        if resolved_creds and os.path.exists(resolved_creds):
            print(f"  {GREEN}✔ Credentials file found at: {resolved_creds}{RESET}")
        else:
            print(f"  {RED}✘ Credentials file NOT found at: {creds_env}{RESET}")
            print(f"    Place your service account JSON key in backend/{creds_env} or provide an absolute path.")
            print(f"    See the guide below for instructions on generating this key.")

    # Step 3: Test BigQuery Client Initialization
    print_step(3, "Initializing BigQuery Client")
    client = get_bigquery_client()
    if not client:
        print(f"  {RED}✘ Failed to initialize BigQuery client.{RESET}")
        print(f"\n{BOLD}Action required:{RESET}")
        print("  1. Create a Service Account in GCP Console (IAM & Admin -> Service Accounts).")
        print("  2. Grant roles: 'BigQuery Data Editor' and 'BigQuery Job User'.")
        print("  3. Create & download a JSON key, place it in backend/gcp-key.json.")
        print("  4. Set GCP_PROJECT_ID and GOOGLE_APPLICATION_CREDENTIALS in backend/.env.")
        return

    print(f"  {GREEN}✔ BigQuery client connected to GCP project: {BOLD}{client.project}{RESET}")

    # Step 4: Ensure Dataset and Table Exist
    print_step(4, f"Verifying BigQuery Dataset '{dataset_name}' & Table '{table_name}'")
    created = ensure_dataset_and_table(client)
    if created:
        print(f"  {GREEN}✔ Dataset '{dataset_name}' and Table '{table_name}' are verified and ready.{RESET}")
    else:
        print(f"  {RED}✘ Failed to verify/create Dataset or Table. Check GCP IAM permissions.{RESET}")
        return

    # Step 5: Test Writing User Log Record
    print_step(5, "Writing Test User Login Record to BigQuery")
    test_user = {
        "sub": "test_google_sub_9999",
        "email": "test.user@finadvisor-demo.com",
        "name": "Test User",
        "picture": "https://lh3.googleusercontent.com/a/default-user"
    }

    success = log_user_to_bigquery(
        user_data=test_user,
        ip_address="127.0.0.1",
        user_agent="BigQueryTestRunner/1.0",
        auth_provider="test_script"
    )

    if success:
        print(f"  {GREEN}✔ Test user login recorded successfully!{RESET}")
    else:
        print(f"  {RED}✘ Failed to record test user login.{RESET}")
        return

    # Step 6: Query Recent Logs to Verify Persistence
    print_step(6, "Reading Recent Login Logs from BigQuery")
    try:
        query = f"""
            SELECT user_id, email, name, auth_provider, login_timestamp, ip_address
            FROM `{client.project}.{dataset_name}.{table_name}`
            ORDER BY login_timestamp DESC
            LIMIT 5
        """
        query_job = client.query(query)
        results = list(query_job.result())

        print(f"\n  Found {len(results)} recent records in BigQuery:")
        print(f"  {'TIMESTAMP':<25} {'EMAIL':<30} {'PROVIDER':<15} {'IP':<15}")
        print(f"  {'-'*25} {'-'*30} {'-'*15} {'-'*15}")
        for row in results:
            ts = str(row.login_timestamp)[:19] if row.login_timestamp else "N/A"
            email = str(row.email or "N/A")[:28]
            provider = str(row.auth_provider or "N/A")[:13]
            ip = str(row.ip_address or "N/A")[:13]
            print(f"  {ts:<25} {email:<30} {provider:<15} {ip:<15}")

        print(f"\n{BOLD}{GREEN}🎉 BigQuery GCP Integration is 100% OPERATIONAL!{RESET}\n")

    except Exception as e:
        print(f"  {YELLOW}⚠ Could not read back rows: {e}{RESET}")
        print("  (Note: insert_rows_json data is streamed into BigQuery immediately, but queryable streaming buffer might take up to a few seconds).")

if __name__ == "__main__":
    main()
