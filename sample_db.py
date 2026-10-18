import sqlite3
from pathlib import Path
from typing import List, Dict, Any


DB_PATH = Path("sample.db")


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def create_database(db_path: Path = DB_PATH) -> None:
    """Create the sample data in the requested SQLite file."""
    connection = get_connection(db_path)
    cursor = connection.cursor()

    cursor.executescript(
        """
        DROP TABLE IF EXISTS orders;
        DROP TABLE IF EXISTS employees;
        DROP TABLE IF EXISTS products;

        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            salary INTEGER NOT NULL,
            department TEXT NOT NULL
        );

        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            price REAL NOT NULL,
            category TEXT NOT NULL
        );

        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            date TEXT NOT NULL,
            FOREIGN KEY (product_id) REFERENCES products(id)
        );
        """
    )

    cursor.executemany(
        "INSERT INTO employees (id, name, salary, department) VALUES (?, ?, ?, ?)",
        [
            (1, "Aarav Sharma", 85000, "Engineering"),
            (2, "Diya Patel", 72000, "Marketing"),
            (3, "Rohan Gupta", 95000, "Engineering"),
            (4, "Meera Nair", 68000, "HR"),
            (5, "Kabir Singh", 78000, "Sales"),
            (6, "Isha Verma", 88000, "Engineering"),
        ],
    )

    cursor.executemany(
        "INSERT INTO products (id, name, price, category) VALUES (?, ?, ?, ?)",
        [
            (1, "Laptop Pro 14", 1299.99, "Electronics"),
            (2, "Noise Cancelling Headphones", 199.50, "Electronics"),
            (3, "Standing Desk", 349.00, "Furniture"),
            (4, "Office Chair", 229.99, "Furniture"),
            (5, "Water Bottle", 24.99, "Accessories"),
            (6, "Mechanical Keyboard", 89.99, "Electronics"),
        ],
    )

    cursor.executemany(
        "INSERT INTO orders (id, product_id, quantity, date) VALUES (?, ?, ?, ?)",
        [
            (1, 1, 2, "2026-01-14"),
            (2, 3, 1, "2026-01-20"),
            (3, 2, 4, "2026-02-05"),
            (4, 6, 3, "2026-02-18"),
            (5, 4, 2, "2026-03-03"),
            (6, 1, 1, "2026-03-11"),
            (7, 5, 10, "2026-03-19"),
        ],
    )

    connection.commit()
    connection.close()


def execute_query(sql: str) -> List[Dict[str, Any]]:
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()
    connection.close()
    return [dict(row) for row in rows]


if __name__ == "__main__":
    create_database()
    print(f"Created sample SQLite database at {DB_PATH.resolve()}")
