import sqlite3
import os
from datetime import datetime
import json

class CharacterDB:
    def __init__(self, db_path="characters.db"):
        self.db_path = db_path
        self.init_db()
        
    def init_db(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create characters table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS characters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT NOT NULL,
            json_data TEXT NOT NULL,
            image_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
        
    def save_character(self, description, json_data, image_path):
        """Save a new character record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO characters (description, json_data, image_path)
        VALUES (?, ?, ?)
        ''', (description, json_data, image_path))
        
        conn.commit()
        conn.close()
        
    def get_all_characters(self):
        """Retrieve all character records"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM characters ORDER BY created_at DESC')
        characters = cursor.fetchall()
        
        conn.close()
        return characters
        
    def get_character_by_id(self, character_id):
        """Retrieve a specific character by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM characters WHERE id = ?', (character_id,))
        character = cursor.fetchone()
        
        conn.close()
        return character

    def display_characters(self):
        """Display all characters in a readable format"""
        characters = self.get_all_characters()
        
        if not characters:
            print("No characters found in the database.")
            return
            
        print("\n=== Character Database ===\n")
        
        for char in characters:
            id, description, json_data, image_path, created_at = char
            print(f"ID: {id}")
            print(f"Created: {created_at}")
            print("\nDescription:")
            print(description)
            print("\nJSON Data:")
            print(json.dumps(json.loads(json_data), indent=2))
            print(f"\nImage Path: {image_path}")
            print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    # Create database instance
    db = CharacterDB()
    
    # Display all characters
    db.display_characters() 