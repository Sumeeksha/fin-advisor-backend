"""
BigQuery Logging Service
Handles writing user authentication & session activity into Google BigQuery database.
Features automatic dataset and table provisioning, robust credential resolution,
and diagnostic health checks.
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
_table_initialized = False


def _resolve_credentials_path() -> Optional[str]:
    """
    Finds the credentials JSON file if specified as a relative path.
    Checks CWD, backend directory, and project root.
    """
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path:
        return None

    if os.path.isabs(creds_path):
        return creds_path if os.path.exists(creds_path) else creds_path

    # Check relative to current working directory
    if os.path.exists(creds_path):
        return os.path.abspath(creds_path)

    # Check relative to backend directory
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cand1 = os.path.join(backend_dir, creds_path)
    if os.path.exists(cand1):
        return cand1

    # Check relative to project root
    project_root = os.path.dirname(backend_dir)
    cand2 = os.path.join(project_root, creds_path)
    if os.path.exists(cand2):
        return cand2

    return creds_path


_last_init_error: Optional[str] = None


def get_bigquery_client():
    """
    Lazy initializer for BigQuery Client.
    Supports GOOGLE_APPLICATION_CREDENTIALS file path, default application credentials,
    and Google Cloud Run environment authentication.
    """
    global _bigquery_client, _last_init_error
    if _bigquery_client is not None:
        return _bigquery_client

    try:
        from google.cloud import bigquery
        from google.oauth2 import service_account

        resolved_creds = _resolve_credentials_path()
        project_id = GCP_PROJECT_ID if (GCP_PROJECT_ID and GCP_PROJECT_ID != "YOUR_GCP_PROJECT_ID") else None

        if resolved_creds and os.path.exists(resolved_creds):
            logger.info(f"Using Google Service Account credentials file: {resolved_creds}")
            credentials = service_account.Credentials.from_service_account_file(resolved_creds)
            target_project = project_id or credentials.project_id
            _bigquery_client = bigquery.Client(credentials=credentials, project=target_project)
        else:
            # If the specified JSON file does NOT exist on disk (e.g. running on Cloud Run),
            # pop GOOGLE_APPLICATION_CREDENTIALS so Google auth falls back cleanly to
            # Application Default Credentials (ADC) / Cloud Run metadata service.
            if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
                os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

            if project_id:
                _bigquery_client = bigquery.Client(project=project_id)
            else:
                _bigquery_client = bigquery.Client()

        logger.info(f"Successfully initialized BigQuery client for project '{_bigquery_client.project}'.")
        _last_init_error = None
        return _bigquery_client

    except ImportError:
        _last_init_error = "google-cloud-bigquery package is not installed."
        logger.error(_last_init_error)
        return None
    except Exception as e:
        _last_init_error = str(e)
        logger.warning(
            f"BigQuery client could not be initialized: {e}. "
            "Login events will be logged to console until GCP credentials are configured."
        )
        return None


def ensure_dataset_and_table(client) -> bool:
    """
    Verifies that the BigQuery dataset and user_logs table exist,
    creating them automatically with the appropriate schema if missing.
    """
    global _table_initialized
    if _table_initialized:
        return True

    try:
        from google.cloud import bigquery
        from google.cloud.exceptions import NotFound

        project_id = client.project
        dataset_id = f"{project_id}.{BIGQUERY_DATASET}"
        table_id = f"{dataset_id}.{BIGQUERY_TABLE}"

        # 1. Check or create dataset
        try:
            client.get_dataset(dataset_id)
        except NotFound:
            logger.info(f"Dataset '{dataset_id}' not found. Creating dataset...")
            dataset = bigquery.Dataset(dataset_id)
            dataset.location = "US"
            client.create_dataset(dataset, timeout=30)
            logger.info(f"Dataset '{dataset_id}' created successfully.")

        # 2. Check or create table
        try:
            client.get_table(table_id)
        except NotFound:
            logger.info(f"Table '{table_id}' not found. Creating table with user logging schema...")
            schema = [
                bigquery.SchemaField("user_id", "STRING", mode="REQUIRED", description="Unique User ID or Google sub"),
                bigquery.SchemaField("email", "STRING", mode="REQUIRED", description="User email address"),
                bigquery.SchemaField("name", "STRING", mode="NULLABLE", description="Display name of user"),
                bigquery.SchemaField("picture", "STRING", mode="NULLABLE", description="Profile picture URL"),
                bigquery.SchemaField("auth_provider", "STRING", mode="NULLABLE", description="Login method (google, email_password, registration)"),
                bigquery.SchemaField("login_timestamp", "TIMESTAMP", mode="REQUIRED", description="UTC timestamp of login event"),
                bigquery.SchemaField("ip_address", "STRING", mode="NULLABLE", description="Client IP address"),
                bigquery.SchemaField("user_agent", "STRING", mode="NULLABLE", description="Browser user agent string"),
            ]
            table = bigquery.Table(table_id, schema=schema)
            # Partition by day on login_timestamp for cost efficiency and fast queries
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="login_timestamp"
            )
            client.create_table(table, timeout=30)
            logger.info(f"Table '{table_id}' created successfully with daily partitioning.")

        _table_initialized = True
        return True
    except Exception as e:
        logger.error(f"Error ensuring BigQuery dataset/table exists: {e}")
        return False


def log_user_to_bigquery(
    user_data: Dict[str, Any],
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    auth_provider: str = "google"
) -> bool:
    """
    Logs user login details into Google BigQuery table.

    Expected Schema for BigQuery Table (`user_logs`):
    - user_id (STRING)
    - email (STRING)
    - name (STRING)
    - picture (STRING)
    - auth_provider (STRING)
    - login_timestamp (TIMESTAMP)
    - ip_address (STRING)
    - user_agent (STRING)
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    log_entry = {
        "user_id": str(user_data.get("sub") or user_data.get("user_id") or "unknown"),
        "email": str(user_data.get("email") or ""),
        "name": str(user_data.get("name") or ""),
        "picture": str(user_data.get("picture") or ""),
        "auth_provider": auth_provider,
        "login_timestamp": timestamp,
        "ip_address": ip_address or "unknown",
        "user_agent": user_agent or "unknown",
    }

    logger.info(
        f"🔑 [USER LOGIN EVENT] Provider: {auth_provider} | "
        f"User: {log_entry['email']} (ID: {log_entry['user_id']}) from IP: {log_entry['ip_address']}"
    )

    client = get_bigquery_client()
    if client:
        try:
            # Ensure dataset and table exist before inserting
            ensure_dataset_and_table(client)

            project_id = client.project
            table_ref = f"{project_id}.{BIGQUERY_DATASET}.{BIGQUERY_TABLE}"
            errors = client.insert_rows_json(table_ref, [log_entry], ignore_unknown_values=True)

            if errors:
                logger.error(f"BigQuery insertion error: {errors}")
                return False

            logger.info(f"✅ Successfully recorded user login in BigQuery table: {table_ref}")
            return True
        except Exception as e:
            logger.error(f"Failed to write log row to BigQuery: {e}")
            return False

    # BigQuery credentials not yet provided
    logger.info("ℹ️ GCP BigQuery not yet configured. Event logged to server output only.")
    return False


def get_bigquery_status() -> Dict[str, Any]:
    """
    Diagnostic helper that checks the state of the BigQuery integration:
    - Checks library installation
    - Checks credentials file existence
    - Tests client connection
    - Reports project, dataset, table status and row count
    """
    status: Dict[str, Any] = {
        "library_installed": False,
        "credentials_file_configured": False,
        "credentials_file_exists": False,
        "credentials_path": None,
        "gcp_project_id": os.getenv("GCP_PROJECT_ID"),
        "dataset_name": BIGQUERY_DATASET,
        "table_name": BIGQUERY_TABLE,
        "connected": False,
        "table_exists": False,
        "total_rows": None,
        "message": ""
    }

    try:
        import google.cloud.bigquery
        status["library_installed"] = True
    except ImportError:
        status["message"] = "google-cloud-bigquery library is not installed."
        return status

    raw_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if raw_creds:
        status["credentials_file_configured"] = True
        resolved = _resolve_credentials_path()
        status["credentials_path"] = resolved
        status["credentials_file_exists"] = bool(resolved and os.path.exists(resolved))

    client = get_bigquery_client()
    if not client:
        err_detail = f" ({_last_init_error})" if _last_init_error else ""
        status["message"] = (
            f"BigQuery client could not connect{err_detail}. "
            "On Cloud Run, ensure the service account has 'BigQuery Data Editor' role. "
            "In local development, provide a valid backend/gcp-key.json file or run: gcloud auth application-default login"
        )
        return status

    status["connected"] = True
    status["active_project"] = client.project

    try:
        from google.cloud.exceptions import NotFound
        table_ref = f"{client.project}.{BIGQUERY_DATASET}.{BIGQUERY_TABLE}"
        table = client.get_table(table_ref)
        status["table_exists"] = True
        status["total_rows"] = table.num_rows
        status["message"] = f"Connected successfully. Table '{table_ref}' has {table.num_rows} logs."
    except NotFound:
        status["table_exists"] = False
        status["message"] = f"Connected to GCP, but dataset or table does not exist yet (will auto-create on first login)."
    except Exception as e:
        status["message"] = f"Connected to GCP, but failed to inspect table: {str(e)}"

    return status
