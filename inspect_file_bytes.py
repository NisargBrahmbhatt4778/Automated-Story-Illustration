# inspect_file_bytes.py
import os

# === Configuration ===
script_dir = os.path.dirname(os.path.abspath(__file__))
embedding_filename = "embeddings.pti" # Make sure this is the file you're having trouble with
embedding_folder = os.path.join(script_dir, "Characters/Milo/trained_models/milo-20250522-011112")
embedding_path = os.path.join(embedding_folder, embedding_filename)
# ===================

if os.path.exists(embedding_path):
    try:
        with open(embedding_path, 'rb') as f:
            header_bytes = f.read(128) # Read first 128 bytes
        print(f"--- File Inspection ---")
        print(f"File: {embedding_path}")
        print(f"Size: {os.path.getsize(embedding_path)} bytes")
        print(f"\nFirst 128 bytes (hex representation):")
        print(header_bytes.hex())
        print(f"\nFirst 128 bytes (printable characters, if any):")
        print("".join(chr(b) if 32 <= b <= 126 else '.' for b in header_bytes))
        print(f"\nFirst 128 bytes (repr output for non-printables):")
        print(repr(header_bytes))

    except Exception as e:
        print(f"Error inspecting file: {e}")
else:
    print(f"File not found: {embedding_path}")