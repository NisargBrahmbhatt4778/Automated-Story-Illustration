import openai
import base64
import requests
import os
from dotenv import load_dotenv
from typing import List, Dict, Tuple
import json

# Load environment variables
load_dotenv()

class CharacterGenerator:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if not self.client.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.chat_history = []
        self.character_name = ""
        self.character_description = ""
        
    def encode_image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    
    def add_to_chat_history(self, role: str, content: List[Dict]):
        self.chat_history.append({"role": role, "content": content})
    
    def initialize_character(self, name: str, description: str):
        """Initialize the character with name and description"""
        self.character_name = name
        self.character_description = description
        
        # Initialize chat history with system message
        self.chat_history = [{
            "role": "system",
            "content": "You are an expert character designer creating consistent children's book-style illustrations. Maintain visual consistency across all generated images for the same character."
        }]
        
        # Add character context
        self.add_to_chat_history("user", [{
            "type": "text",
            "text": f"Creating character sheets for: {name}\nDescription: {description}"
        }])
    
    def generate_image(self, prompt: str, size: Tuple[int, int] = (1024, 1024)) -> str:
        """Generate an image using OpenAI's image generation API"""
        try:
            # Add the current request to chat history
            self.add_to_chat_history("user", [{
                "type": "text",
                "text": prompt
            }])
            
            # Create a more detailed prompt that includes previous context
            context_prompt = "Maintain consistent style with previous generations. "
            if len(self.chat_history) > 2:  # If we have previous generations
                context_prompt += "Reference the previous character designs and maintain the same art style, proportions, and character details. "
            
            full_prompt = f"{context_prompt}\n\n{prompt}"
            
            response = self.client.images.generate(
                model="gpt-image-1",
                prompt=full_prompt,
                n=1,
                size=f"{size[0]}x{size[1]}",
                quality="hd",
                style="natural"
            )
            
            image_url = response.data[0].url
            
            # Add the response to chat history
            self.add_to_chat_history("assistant", [{
                "type": "image_url",
                "image_url": {"url": image_url}
            }])
            
            return image_url
            
        except Exception as e:
            print(f"Error generating image: {str(e)}")
            return None
    
    def generate_sheet(self, template_path: str, prompt_path: str, output_path: str, resolution: Tuple[int, int]) -> bool:
        """Generate a single character sheet"""
        try:
            # Read the prompt template
            with open(prompt_path, "r") as f:
                prompt_template = f.read()
            
            # Format the prompt with character details and previous context
            base_prompt = prompt_template.format(
                Char_Name=self.character_name,
                Char_Description=self.character_description
            )
            
            # Add the grid template to chat history
            template_base64 = self.encode_image_to_base64(template_path)
            self.add_to_chat_history("user", [
                {"type": "text", "text": "Here is the grid template to follow for the layout:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{template_base64}"}}
            ])
            
            # Add reference to previous generations if they exist
            previous_sheets = [msg["content"][0]["image_url"]["url"] 
                             for msg in self.chat_history 
                             if msg["role"] == "assistant" 
                             and "image_url" in msg["content"][0]
                             and msg["content"][0].get("type") == "generated_sheet"]
            
            if previous_sheets:
                base_prompt += "\n\nMaintain consistency with the previously generated character sheets, keeping the same art style, character proportions, and design elements. Follow the grid layout exactly as shown in the template."
            
            # Generate the image
            image_url = self.generate_image(base_prompt, resolution)
            
            if image_url:
                # Download and save the image
                response = requests.get(image_url)
                with open(output_path, "wb") as out_file:
                    out_file.write(response.content)
                print(f"✅ Successfully generated {output_path}")
                
                # Add the generated sheet to chat history with metadata
                self.add_to_chat_history("assistant", [{
                    "type": "generated_sheet",  # Add type to distinguish from other images
                    "sheet_type": os.path.basename(output_path).split('_')[1].split('.')[0],  # Extract sheet type
                    "image_url": {"url": image_url}
                }])
                
                # Add descriptive context about the generated sheet
                self.add_to_chat_history("user", [{
                    "type": "text",
                    "text": f"This is the {os.path.basename(output_path)} showing the character's design. Use this as reference for maintaining consistency in future generations. Make sure to follow the same grid layout structure."
                }])
                
                return True
            else:
                print(f"❌ Failed to generate {output_path}")
                return False
                
        except Exception as e:
            print(f"❌ Error generating {output_path}: {str(e)}")
            return False
    
    def generate_all_sheets(self, output_dir: str = "Characters") -> Dict[str, bool]:
        """Generate all character sheets"""
        # Create character-specific directory structure
        char_dir = os.path.join(output_dir, self.character_name, "uncut_images")
        os.makedirs(char_dir, exist_ok=True)
        
        # Define the tasks
        tasks = [
            {
                "name": "character_sheet",
                "template": "z_GPT_Templates/grid_with_sections.png",
                "prompt": "z_GPT_Templates/GPT_Template_for_Sheet_Gen.txt",
                "output": os.path.join(char_dir, f"{self.character_name}_character_sheet.png"),
                "resolution": (1536, 1024)
            },
            {
                "name": "action_sheet",
                "template": "z_GPT_Templates/grid_with_sections.png",
                "prompt": "z_GPT_Templates/GPT_Template_for_Action_Sheet_Gen.txt",
                "output": os.path.join(char_dir, f"{self.character_name}_action_sheet.png"),
                "resolution": (1536, 1024)
            },
            {
                "name": "emotion_sheet",
                "template": "z_GPT_Templates/grid_with_sections_square.png",
                "prompt": "z_GPT_Templates/GPT_Template_for_Emotion_Sheet_Gen.txt",
                "output": os.path.join(char_dir, f"{self.character_name}_emotion_sheet.png"),
                "resolution": (1024, 1024)
            }
        ]
        
        results = {}
        for task in tasks:
            print(f"\nGenerating {task['name']}...")
            success = self.generate_sheet(
                task["template"],
                task["prompt"],
                task["output"],
                task["resolution"]
            )
            results[task["name"]] = success
            
            # If generation was successful, add context for next generation
            if success:
                self.add_to_chat_history("user", [{
                    "type": "text",
                    "text": f"Here is the generated {task['name']} for reference in maintaining visual consistency."
                }])
        
        return results

def main():
    # Example usage
    generator = CharacterGenerator()
    
    # Get character details from user
    character_name = input("Enter character name: ")
    character_description = input("Enter character description: ")
    
    # Initialize the generator with character details
    generator.initialize_character(character_name, character_description)
    
    # Generate all sheets
    results = generator.generate_all_sheets()
    
    # Print results
    print("\nGeneration Results:")
    for sheet_type, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"{sheet_type}: {status}")

if __name__ == "__main__":
    main() 