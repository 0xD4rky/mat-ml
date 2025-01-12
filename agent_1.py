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