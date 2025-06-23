import os
import sqlite3

DATABASE_PATH = os.path.join(os.path.dirname(__file__), "database.db")


def create_database():
    # Comprobar si la base de datos ya existe
    db_exists = os.path.exists(DATABASE_PATH)

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    if not db_exists:
        print(f"Creando nueva base de datos en {DATABASE_PATH}...")
        # Crear tabla de usuarios
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT NOT NULL
            )
        """
        )
        print("Tabla 'users' creada.")

        # Crear tabla de tareas (todos)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS todos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                task TEXT NOT NULL,
                completed BOOLEAN NOT NULL DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """
        )
        print("Tabla 'todos' creada.")

        # Insertar usuarios de ejemplo
        dummy_users = [
            ("alice", "alice@example.com"),
            ("bob", "bob@example.com"),
            ("charlie", "charlie@example.com"),
        ]
        cursor.executemany(
            "INSERT INTO users (username, email) VALUES (?, ?)", dummy_users
        )
        print(f"Insertados {len(dummy_users)} usuarios de ejemplo.")

        # Insertar tareas de ejemplo
        dummy_todos = [
            (1, "Comprar víveres", 0),
            (1, "Leer un libro", 1),
            (2, "Terminar informe del proyecto", 0),
            (2, "Salir a correr", 0),
            (3, "Planificar viaje de fin de semana", 1),
        ]
        cursor.executemany(
            "INSERT INTO todos (user_id, task, completed) VALUES (?, ?, ?)", dummy_todos
        )
        print(f"Insertadas {len(dummy_todos)} tareas de ejemplo.")

        conn.commit()
        print("Base de datos creada y poblada exitosamente.")
    else:
        print(f"La base de datos ya existe en {DATABASE_PATH}. No se realizaron cambios.")

    conn.close()


if __name__ == "__main__":
    create_database()
