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

    parser.add_argument('path_tflite', nargs='?', default='../tflite/tflite_model.tflite')
    parser.add_argument('path_cpp_file', nargs='?', default='../tflite/airway_tflite/src/model.cpp')
    args = parser.parse_args()

    if not os.path.exists(args.path_tflite):
        raise ValueError(f"Given TFLite file path '{args.path_tflite}' does not exist.")
    if not os.path.exists(args.path_cpp_file):
        raise ValueError(f"Given CPP file path '{args.path_cpp_file}' does not exist.")

    path_c_array = args.path_tflite.replace(".tflite", ".cc")

    convert_tflite_to_c_array(args.path_tflite, path_c_array)
    insert_c_array_into_cpp_file(path_c_array, args.path_cpp_file)