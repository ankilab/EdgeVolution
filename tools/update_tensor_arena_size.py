import argparse
import re

def update_tensor_arena_size(file_path, new_size_kb):
    """
    Update the kTensorArenaSize in the given .cpp file.
    
    Args:
        file_path (str): Path to the main.cpp file.
        new_size_kb (int): New size for kTensorArenaSize in KB.
    
    Returns:
        bool: True if modification was successful, False otherwise.
    """
    try:
        # Read the file
        with open(file_path, "r") as file:
            content = file.readlines()

        # Regex pattern to find and replace kTensorArenaSize definition
        pattern = re.compile(r"(constexpr\s+int\s+kTensorArenaSize\s*=\s*)\d+\s*\*\s*1024\s*;")

        modified = False
        for i, line in enumerate(content):
            if pattern.search(line):
                # Replace with the new value
                content[i] = f"    constexpr int kTensorArenaSize = {new_size_kb} * 1024;\n"
                modified = True
                break

        if not modified:
            print("kTensorArenaSize definition not found in the file.")
            return False

        # Write the modified content back to the file
        with open(file_path, "w") as file:
            file.writelines(content)

        print(f"Successfully updated kTensorArenaSize to {new_size_kb} KB.")
        return True

    except Exception as e:
        print(f"Error updating kTensorArenaSize: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update kTensorArenaSize in main.cpp file.")
    parser.add_argument("file", type=str, help="Path to the .cpp file.")
    parser.add_argument("size", type=int, help="New size for kTensorArenaSize in KB (means this number will be multiplied by 1024).")
    args = parser.parse_args()

    update_tensor_arena_size(args.file, args.size)