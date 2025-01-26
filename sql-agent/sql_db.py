import sqlite3
import os
from datetime import datetime, timedelta

def create_sample_database():
    if os.path.exists("test_company.db"):
        os.remove("test_company.db")
    
    conn = sqlite3.connect("test_company.db")
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE employees (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        department TEXT NOT NULL,
        salary REAL NOT NULL,
        hire_date DATE NOT NULL
    )""")
    
    cursor.execute("""
    CREATE TABLE departments (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        budget REAL NOT NULL,
        location TEXT NOT NULL
    )""")
    
    cursor.execute("""
    CREATE TABLE projects (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        start_date DATE,
        end_date DATE,
        department_id INTEGER,
        FOREIGN KEY (department_id) REFERENCES departments (id)
    )""")
    
    departments_data = [
        (1, "Engineering", 1000000, "New York"),
        (2, "Marketing", 500000, "San Francisco"),
        (3, "HR", 300000, "Chicago"),
        (4, "Sales", 800000, "Boston")
    ]
    
    employees_data = [
        (1, "John Smith", "Engineering", 85000, "2023-01-15"),
        (2, "Sarah Johnson", "Marketing", 65000, "2023-02-01"),
        (3, "Michael Brown", "Engineering", 90000, "2023-03-10"),
        (4, "Emily Davis", "HR", 55000, "2023-04-20"),
        (5, "David Wilson", "Sales", 75000, "2023-05-05"),
        (6, "Lisa Anderson", "Engineering", 95000, "2023-06-15"),
        (7, "James Taylor", "Marketing", 70000, "2023-07-01"),
        (8, "Emma Martinez", "Sales", 80000, "2023-08-10")
    ]
    
    projects_data = [
        (1, "Website Redesign", "2023-02-01", "2023-06-30", 1),
        (2, "Q4 Marketing Campaign", "2023-10-01", "2023-12-31", 2),
        (3, "Employee Training Program", "2023-03-15", "2023-09-30", 3),
        (4, "Sales Analytics Platform", "2023-05-01", "2023-12-31", 4)
    ]
    
    cursor.executemany("INSERT INTO departments VALUES (?, ?, ?, ?)", departments_data)
    cursor.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?)", employees_data)
    cursor.executemany("INSERT INTO projects VALUES (?, ?, ?, ?, ?)", projects_data)
    
    conn.commit()
    conn.close()
    
    print("Sample database created successfully!")
    return "test_company.db"

if __name__ == "__main__":
    db_path = create_sample_database()