import pandas as pd
from sqlalchemy import create_engine, inspect, text


class DBService:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.engine = create_engine(connection_string)

    def test_connection(self):
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))

    def get_schema_info(self) -> dict:
        inspector = inspect(self.engine)
        tables = inspector.get_table_names()

        schema = {}
        for table in tables:
            columns = inspector.get_columns(table)
            schema[table] = [
                {"name": col["name"], "type": str(col["type"])}
                for col in columns
            ]
        return schema

    @staticmethod
    def schema_to_text(schema_info: dict) -> str:
        lines = []
        for table, columns in schema_info.items():
            lines.append(f"Table: {table}")
            for col in columns:
                lines.append(f"  - {col['name']} ({col['type']})")
        return "\n".join(lines)

    def run_select_query(self, sql: str) -> pd.DataFrame:
        with self.engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)
        return df