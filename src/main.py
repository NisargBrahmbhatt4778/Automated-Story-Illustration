"""
This is the main file for the storybook character generator.

It will run the differnt stages of the charecter gen pipleine. The stagesa are as follows:

1. Get a character name + Description description from the user
    1a. Filter the charecter description to remove the charecter name and make sure everything in the description only 
2. Generate a character sheet and an action sheet for the character (manual step)
    2a. Using the name and description, create a prompt to give to ChatGPT to generate the character sheet and action sheet.
3. The user places all the sheets in a folder names the character name
4. Run some sort of code that then cuts up the charecter sheet into separate charecter images
5. Upscale the cut images using Real-ESRGAN (Anime6B) for better quality
6. Call the API to train the SDXL model on the charecter images.
7. Using the trained model, ask it to generate another image using the original character description.
8. Iterative step for the user to keep asking for new images of the same charecter.


Step 6 to be expanded and made much better in the future.

"""

from sheet_handler import handle_character_sheets
from image_processor import process_character_images
from upscaler import upscale_character_images

# Step 1 - Get Character Name + Description
def get_character_info():
    print("\n=== Character Information Input ===")
    character_name = input("Enter the character's name: ").strip()
    while not character_name:
        print("Character name cannot be empty!")
        character_name = input("Enter the character's name: ").strip()
    
    print("\nPlease provide a detailed description of the character.")
    print("Include details like:")
    print("- Physical appearance (height, build, hair color, etc.)")
    print("- Clothing and accessories")
    print("- Notable features or characteristics")
    print("- Any specific style preferences")
    
    character_description = input("\nEnter character description: ").strip()
    while not character_description:
        print("Character description cannot be empty!")
        character_description = input("Enter character description: ").strip()
    
    return character_name, character_description

# Step 4 - The 3 images need to be sliced and put into a folder called 'cut_images'
def process_images(character_name, uncut_dir):
    """
    Process the character images by cutting them into sections.
    
    Args:
        character_name (str): Name of the character
        uncut_dir (Path): Path to the directory containing uncut images
    """
    print("\n=== Processing Character Images ===")
    cut_dir = process_character_images(character_name, uncut_dir)
    if cut_dir:
        print("Image processing completed successfully!")
        return cut_dir
    else:
        print("Error processing images!")
        return None

if __name__ == "__main__":
    character_name, character_description = get_character_info()
    print(f"\nCharacter Name: {character_name}")
    print(f"Character Description: {character_description}")
    
    # Handle the character sheet uploads sequentially
    uncut_dir = handle_character_sheets(character_name, character_description)
    
    if uncut_dir:
        # Process the images
        cut_dir = process_images(character_name, uncut_dir)
        
        if cut_dir:
            # Upscale the processed images
            upscaled_dir = upscale_character_images(character_name, cut_dir)
            if upscaled_dir:
                print(f"\nUpscaled images saved to: {upscaled_dir}")
            else:
                print("\nWarning: Image upscaling failed or was incomplete.")
