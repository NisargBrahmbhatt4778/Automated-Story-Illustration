"""
This module handles storing and retrieving the history of generated images.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
import os
import shutil

class GenerationHistory:
    def __init__(self, db_path="generation_history.db"):
        """
        Initialize the generation history database.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        """Initialize the database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create generations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS generations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    character_name TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    image_paths TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def save_generation(self, character_name, model_id, prompt, parameters, image_urls):
        """
        Save a new generation to the database and download the images.
        
        Args:
            character_name (str): Name of the character
            model_id (str): ID of the model used
            prompt (str): The prompt used for generation
            parameters (dict): The parameters used for generation
            image_urls (list): List of URLs of the generated images
            
        Returns:
            int: ID of the saved generation
        """
        # Create directory for storing images if it doesn't exist
        base_dir = Path("generated_images") / character_name
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Download and save images
        image_paths = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, url in enumerate(image_urls):
            # Create a unique filename for each image
            filename = f"gen_{timestamp}_{i+1}.png"
            image_path = base_dir / filename
            
            # Download the image (you'll need to implement this based on your needs)
            # For now, we'll just store the URL
            image_paths.append(str(image_path))
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO generations 
                (character_name, model_id, prompt, parameters, image_paths)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                character_name,
                model_id,
                prompt,
                json.dumps(parameters),
                json.dumps(image_paths)
            ))
            generation_id = cursor.lastrowid
            conn.commit()
        
        return generation_id
    
    def get_generations(self, character_name=None, limit=10):
        """
        Retrieve generation history.
        
        Args:
            character_name (str, optional): Filter by character name
            limit (int): Maximum number of records to return
            
        Returns:
            list: List of generation records
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if character_name:
                cursor.execute('''
                    SELECT * FROM generations 
                    WHERE character_name = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (character_name, limit))
            else:
                cursor.execute('''
                    SELECT * FROM generations 
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (limit,))
            
            columns = [description[0] for description in cursor.description]
            results = []
            
            for row in cursor.fetchall():
                record = dict(zip(columns, row))
                record['parameters'] = json.loads(record['parameters'])
                record['image_paths'] = json.loads(record['image_paths'])
                results.append(record)
            
            return results
    
    def get_generation(self, generation_id):
        """
        Retrieve a specific generation by ID.
        
        Args:
            generation_id (int): ID of the generation to retrieve
            
        Returns:
            dict: Generation record or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM generations 
                WHERE id = ?
            ''', (generation_id,))
            
            row = cursor.fetchone()
            if row:
                columns = [description[0] for description in cursor.description]
                record = dict(zip(columns, row))
                record['parameters'] = json.loads(record['parameters'])
                record['image_paths'] = json.loads(record['image_paths'])
                return record
            return None
    
    def delete_generation(self, generation_id):
        """
        Delete a generation and its associated images.
        
        Args:
            generation_id (int): ID of the generation to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        # First get the generation record
        generation = self.get_generation(generation_id)
        if not generation:
            return False
        
        # Delete the images
        for image_path in generation['image_paths']:
            try:
                os.remove(image_path)
            except OSError:
                pass  # Ignore if file doesn't exist
        
        # Delete from database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM generations 
                WHERE id = ?
            ''', (generation_id,))
            conn.commit()
        
        return True 