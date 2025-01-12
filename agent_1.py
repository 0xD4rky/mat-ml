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
    
    def execute_query(self, query: str) -> list:
        try:
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            column_names = [description[0] for description in self.cursor.description]
            formatted_results = [dict(zip(column_names, row)) for row in results]
            return formatted_results
        except Exception as e:
            return [{"error": str(e)}]
            
    def process_natural_query(self, user_query: str) -> tuple[str, list]:
        sql_query = self.generate_sql(user_query)
        results = self.execute_query(sql_query)
        return sql_query, results
        
    def close(self):
        self.conn.close()


