import os
from sqlalchemy import create_engine, text
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class MySQLClient:
    """Manages direct read-only SQL execution on the Nikky Frozen MySQL Database."""
    
    _engine = None

    @classmethod
    def get_engine(cls):
        if cls._engine is None:
            # Attempt to read Laravel .env for DB credentials
            db_user = "root"
            db_pass = ""
            db_host = "127.0.0.1"
            db_port = "3306"
            db_name = "pos_nikky_frozen"

            laravel_env_path = os.path.join(os.path.dirname(__file__), "..", "..", "backend", ".env")
            if os.path.exists(laravel_env_path):
                try:
                    with open(laravel_env_path, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith("#"):
                                continue
                            if "=" in line:
                                k, v = line.split("=", 1)
                                k = k.strip()
                                v = v.strip().strip('"').strip("'")
                                if k == "DB_USERNAME":
                                    db_user = v
                                elif k == "DB_PASSWORD":
                                    db_pass = v
                                elif k == "DB_HOST":
                                    db_host = v
                                elif k == "DB_PORT":
                                    db_port = v
                                elif k == "DB_DATABASE":
                                    db_name = v
                except Exception as e:
                    logger.error(f"Failed to read Laravel .env: {e}")

            # Connection URI using pymysql
            connection_uri = f"mysql+pymysql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
            try:
                cls._engine = create_engine(connection_uri)
                logger.info("MySQL Database engine initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to create SQLAlchemy engine: {e}")
                raise e

        return cls._engine

    @classmethod
    def execute_query(cls, query: str, params: dict = None) -> list:
        """Executes read-only SQL query and returns list of dicts.
        Returns empty list with a warning if MySQL is not reachable.
        """
        try:
            engine = cls.get_engine()
            with engine.connect() as conn:
                clean_query = query.strip().lower()
                if not clean_query.startswith("select") and not clean_query.startswith("show") and not clean_query.startswith("desc"):
                    raise PermissionError("Only SELECT or read-only queries are allowed.")
                result = conn.execute(text(query), params or {})
                return [dict(row._mapping) for row in result]
        except Exception as e:
            logger.warning(f"MySQL not reachable or query failed (returning []): {e}")
            return []

    @classmethod
    def execute_query_df(cls, query: str, params: dict = None) -> pd.DataFrame:
        """Executes query and returns a Pandas DataFrame."""
        engine = cls.get_engine()
        # Enforce read-only check
        clean_query = query.strip().lower()
        if not clean_query.startswith("select") and not clean_query.startswith("show") and not clean_query.startswith("desc"):
            raise PermissionError("Only SELECT or read-only queries are allowed.")
        
        return pd.read_sql(text(query), engine, params=params or {})
