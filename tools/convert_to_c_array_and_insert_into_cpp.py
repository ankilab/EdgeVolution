import argparse
import os
import subprocess


def convert_tflite_to_c_array(path_tflite, path_c_array):
    subprocess.call("xxd -i " + path_tflite + " > " + path_c_array, shell=True)


def insert_c_array_into_cpp_file(path_c_array, path_cpp_file):
    # read .cpp and .cc file
    with open(path_cpp_file, 'r') as cpp_file, open(path_c_array, 'r') as cc_file:
        cpp_lines = cpp_file.readlines()
        cc_lines = cc_file.readlines()

    start_c_array = 2  # C-array in .cpp file starts at index 15
    with open(path_cpp_file, 'w') as cpp_file:
        # add header to cpp file
        cpp_file.write('#include "model.hpp" \n')
        cpp_file.write('alignas(8) const unsigned char g_model[] = {\n')

        # delete whole C-array from .cpp file
        for i in range(start_c_array, len(cpp_lines)):
            cpp_file.write("")

        # write new C-array to .cpp file
        for cc_line in cc_lines[1::]:
            cc_line = cc_line.replace("unsigned int ___tflite_tflite_model_tflite_len", "const int g_model_len")
            cpp_file.write(cc_line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='ConverTFLiteToCArray',
        description='This script converts a TFLite model to a C array and inserts it into the cpp file.')

    parser.add_argument('path_tflite', nargs='?', default='None')
    parser.add_argument('path_cpp_file', nargs='?', default='../tflite/edgevolution_tflite/src/model.cpp')
    args = parser.parse_args()

    def find_project_root():
        """Find the EdgeVolution project root directory."""
        current_dir = os.getcwd()
        
        # Check if current directory is EdgeVolution
        if os.path.basename(current_dir).lower() == "edgevolution":
            return current_dir
            
        # Try up to 3 levels up in the directory hierarchy
        for _ in range(3):
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:  # Reached filesystem root
                break
                
            current_dir = parent_dir
            # Check if this directory is EdgeVolution
            if os.path.basename(current_dir).lower() == "edgevolution":
                return current_dir
                
            # Check subdirectories
            try:
                for item in os.listdir(current_dir):
                    item_path = os.path.join(current_dir, item)
                    if os.path.isdir(item_path) and item.lower() == "edgevolution":
                        return item_path
            except (PermissionError, FileNotFoundError):
                pass
        
        return None

    # Check if paths exist
    if not os.path.exists(args.path_tflite) or not os.path.exists(args.path_cpp_file):
        project_root = find_project_root()
        
        if project_root:
            print(f"Found EdgeVolution root at: {project_root}")
            
            if not os.path.exists(args.path_tflite):
                adjusted_path = os.path.join(project_root, args.path_tflite.lstrip('/'))
                if os.path.exists(adjusted_path):
                    args.path_tflite = adjusted_path
                    print(f"Found TFLite file at: {args.path_tflite}")
            
            if not os.path.exists(args.path_cpp_file):
                adjusted_path = os.path.join(project_root, args.path_cpp_file.lstrip('/'))
                if os.path.exists(adjusted_path):
                    args.path_cpp_file = adjusted_path
                    print(f"Found CPP file at: {args.path_cpp_file}")
                else:
                    default_path = os.path.join(project_root, "tflite/edgevolution_tflite/src/model.cpp")
                    if os.path.exists(default_path):
                        args.path_cpp_file = default_path
                        print(f"Using default CPP file: {args.path_cpp_file}")
    
    # Final check
    if not os.path.exists(args.path_tflite):
        print(f"Current directory: {os.getcwd()}")
        raise ValueError(f"TFLite file not found: '{args.path_tflite}'. Make sure you're in the EdgeVolution root directory.")
    
    if not os.path.exists(args.path_cpp_file):
        print(f"Current directory: {os.getcwd()}")
        raise ValueError(f"CPP file not found: '{args.path_cpp_file}'. Make sure you're in the EdgeVolution root directory.")

    path_c_array = args.path_tflite.replace(".tflite", ".cc")

    convert_tflite_to_c_array(args.path_tflite, path_c_array)
    insert_c_array_into_cpp_file(path_c_array, args.path_cpp_file)