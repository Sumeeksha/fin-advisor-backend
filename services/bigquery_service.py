"""
BigQuery Logging Service
Handles writing user authentication & session activity into Google BigQuery database.
"""

import os
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

logger = logging.getLogger("bigquery_service")
logger.setLevel(logging.INFO)

# BigQuery Dataset and Table configuration
BIGQUERY_DATASET = os.getenv("BIGQUERY_DATASET", "financial_app_db")
BIGQUERY_TABLE = os.getenv("BIGQUERY_TABLE", "user_logs")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", None)

_bigquery_client = None

def get_bigquery_client():
    """
    Lazy initializer for BigQuery Client.
    Will initialize using environment variable GOOGLE_APPLICATION_CREDENTIALS
    or default service account credentials if available.
    """
    global _bigquery_client
    if _bigquery_client is not None:
        return _bigquery_client

    try:
        from google.cloud import bigquery
        # Initialize client (uses GOOGLE_APPLICATION_CREDENTIALS env var or GCP environment)
        if GCP_PROJECT_ID:
            _bigquery_client = bigquery.Client(project=GCP_PROJECT_ID)
        else:
            _bigquery_client = bigquery.Client()
        logger.info("Successfully initialized BigQuery client.")
        return _bigquery_client
    except Exception as e:
        logger.warning(
            f"BigQuery client could not be initialized (gcp credentials pending): {e}. "
            "User login events will be logged to standard output until GCP credentials are configured."
        )
        return None


def log_user_to_bigquery(
    user_data: Dict[str, Any],
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None
) -> bool:
    """
    Logs user login details into Google BigQuery table.

    Expected Schema for BigQuery Table (`user_logs`):
    - user_id (STRING) -> Google 'sub'
    - email (STRING)
    - name (STRING)
    - picture (STRING)
    - login_timestamp (TIMESTAMP)
    - ip_address (STRING)
    - user_agent (STRING)
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    
    log_entry = {
        "user_id": user_data.get("sub"),
        "email": user_data.get("email"),
        "name": user_data.get("name"),
        "picture": user_data.get("picture"),
        "login_timestamp": timestamp,
        "ip_address": ip_address or "unknown",
        "user_agent": user_agent or "unknown"
    }

    logger.info(f"🔑 [USER LOGIN EVENT] User: {log_entry['email']} (ID: {log_entry['user_id']}) from IP: {log_entry['ip_address']}")

    # Try BigQuery insert if client is available
    client = get_bigquery_client()
    if client:
        try:
            project_id = client.project
            table_ref = f"{project_id}.{BIGQUERY_DATASET}.{BIGQUERY_TABLE}"
            errors = client.insert_rows_json(table_ref, [log_entry])
            
            if errors:
                logger.error(f"BigQuery insertion error: {errors}")
                return False
            
            logger.info(f"Successfully recorded user login in BigQuery table: {table_ref}")
            return True
        except Exception as e:
            logger.error(f"Failed to write log row to BigQuery: {e}")
            return False

    # Fallback when BigQuery client is not configured yet
    return True
