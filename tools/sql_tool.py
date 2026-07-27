import logging
from database.mysql_client import MySQLClient

logger = logging.getLogger(__name__)

class SQLTool:
    """Wrapper class for executing read-only SQL queries on the POS database."""
    
    @staticmethod
    def execute(query: str, params: dict = None) -> list:
        clean_query = query.strip().lower()
        if not clean_query.startswith("select") and not clean_query.startswith("show") and not clean_query.startswith("desc"):
            logger.warning(f"Blocked unsafe SQL query: {query}")
            raise PermissionError("Only SELECT or read-only queries are allowed.")
            
        try:
            logger.info(f"Executing SQL via SQLTool: {query}")
            return MySQLClient.execute_query(query, params)
        except Exception as e:
            logger.error(f"SQLTool execution error: {e}")
            raise e
