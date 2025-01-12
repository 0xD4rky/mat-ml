import google.generativeai as genai
import sqlite3
from typing import Optional
import os

class SQLAgent:
    def __init__(self, api_key: str, database_path: str):
        self.database_path = database_path
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.conn = sqlite3.connect(database_path)
        self.cursor = self.conn.cursor()

    def get_schema(self) -> str:
        schema_query = """
        SELECT name, sql FROM sqlite_master 
        WHERE type='table';
        """
        self.cursor.execute(schema_query)
        schema = []
        for table_name, table_sql in self.cursor.fetchall():
            schema.append(f"Table: {table_name}")
            cols_query = f"PRAGMA table_info({table_name});"
            self.cursor.execute(cols_query)
            columns = self.cursor.fetchall()
            for col in columns:
                schema.append(f"- {col[1]} ({col[2]})")
        return "\n".join(schema)
    
    def generate_sql(self, user_query: str) -> str:
        schema = self.get_schema()
        prompt = f"""
        Given the following database schema:

        {schema}

        Convert this natural language query to SQL:
        {user_query}

        Respond with ONLY the SQL query, nothing else."""

        response = self.model.generate_content(prompt)
        return response.text.strip()

