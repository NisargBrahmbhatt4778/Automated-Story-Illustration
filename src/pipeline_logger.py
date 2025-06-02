"""
Centralized pipeline logging utility for the character generation pipeline.
This module provides logging functionality that spans the entire pipeline process.
"""

import os
import logging
from datetime import datetime
from pathlib import Path

class PipelineLogger:
    """Centralized logger for the entire character generation pipeline"""
    
    def __init__(self, character_name):
        self.character_name = character_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"img_gen_logs/pipeline_{character_name}_{self.timestamp}.log"
        
        # Create img_gen_logs directory if it doesn't exist
        os.makedirs("img_gen_logs", exist_ok=True)
        
        # Setup logger
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Setup the pipeline logger with file and console handlers"""
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
        
        # Create logger
        logger = logging.getLogger(f'pipeline_{self.character_name}')
        logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Create file handler
        file_handler = logging.FileHandler(self.log_filename)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Create console handler with simpler format
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_pipeline_start(self, character_description):
        """Log the start of the pipeline"""
        self.logger.info("=" * 80)
        self.logger.info("STARTING CHARACTER GENERATION PIPELINE")
        self.logger.info("=" * 80)
        self.logger.info(f"Character Name: {self.character_name}")
        self.logger.info(f"Character Description: {character_description}")
        self.logger.info(f"Timestamp: {self.timestamp}")
        self.logger.info(f"Log File: {self.log_filename}")
        self.logger.info("=" * 80)
    
    def log_stage_start(self, stage_number, stage_name, stage_description=""):
        """Log the start of a pipeline stage"""
        self.logger.info("")
        self.logger.info(f"STAGE {stage_number}: {stage_name}")
        self.logger.info("-" * 50)
        if stage_description:
            self.logger.info(f"Description: {stage_description}")
    
    def log_stage_success(self, stage_number, stage_name, details=""):
        """Log successful completion of a pipeline stage"""
        self.logger.info(f"✓ STAGE {stage_number} COMPLETED: {stage_name}")
        if details:
            self.logger.info(f"Details: {details}")
        self.logger.info("")
    
    def log_stage_error(self, stage_number, stage_name, error_message):
        """Log error in a pipeline stage"""
        self.logger.error(f"✗ STAGE {stage_number} FAILED: {stage_name}")
        self.logger.error(f"Error: {error_message}")
        self.logger.error("")
    
    def log_info(self, message):
        """Log an info message"""
        self.logger.info(message)
    
    def log_warning(self, message):
        """Log a warning message"""
        self.logger.warning(message)
    
    def log_error(self, message):
        """Log an error message"""
        self.logger.error(message)
    
    def log_sheet_generation(self, sheet_type, status, details=""):
        """Log character sheet generation progress"""
        if status == "start":
            self.logger.info(f"  → Generating {sheet_type} sheet...")
        elif status == "success":
            self.logger.info(f"  ✓ {sheet_type} sheet generated successfully")
            if details:
                self.logger.info(f"    Saved to: {details}")
        elif status == "error":
            self.logger.error(f"  ✗ Failed to generate {sheet_type} sheet")
            if details:
                self.logger.error(f"    Error: {details}")
    
    def log_file_operation(self, operation, file_path, status, details=""):
        """Log file operations (save, load, process, etc.)"""
        if status == "start":
            self.logger.info(f"  → {operation}: {file_path}")
        elif status == "success":
            self.logger.info(f"  ✓ {operation} successful: {file_path}")
            if details:
                self.logger.info(f"    {details}")
        elif status == "error":
            self.logger.error(f"  ✗ {operation} failed: {file_path}")
            if details:
                self.logger.error(f"    Error: {details}")
    
    def log_api_call(self, api_name, operation, status, details=""):
        """Log API calls and responses"""
        if status == "start":
            self.logger.info(f"  → {api_name} API: {operation}")
        elif status == "success":
            self.logger.info(f"  ✓ {api_name} API call successful: {operation}")
            if details:
                self.logger.info(f"    {details}")
        elif status == "error":
            self.logger.error(f"  ✗ {api_name} API call failed: {operation}")
            if details:
                self.logger.error(f"    Error: {details}")
    
    def log_pipeline_complete(self, success=True, final_message=""):
        """Log the completion of the pipeline"""
        self.logger.info("")
        self.logger.info("=" * 80)
        if success:
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        else:
            self.logger.error("PIPELINE FAILED!")
        
        if final_message:
            self.logger.info(final_message)
        
        self.logger.info(f"Total execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Full log saved to: {self.log_filename}")
        self.logger.info("=" * 80)
    
    def get_log_filename(self):
        """Get the log filename"""
        return self.log_filename
    
    def get_logger(self):
        """Get the underlying logger object"""
        return self.logger

# Global pipeline logger instance
_pipeline_logger = None

def get_pipeline_logger():
    """Get the current pipeline logger instance"""
    global _pipeline_logger
    return _pipeline_logger

def init_pipeline_logger(character_name):
    """Initialize the pipeline logger for a character"""
    global _pipeline_logger
    _pipeline_logger = PipelineLogger(character_name)
    return _pipeline_logger

def cleanup_pipeline_logger():
    """Cleanup the pipeline logger"""
    global _pipeline_logger
    if _pipeline_logger:
        # Close handlers to release file locks
        for handler in _pipeline_logger.logger.handlers:
            handler.close()
        _pipeline_logger = None
