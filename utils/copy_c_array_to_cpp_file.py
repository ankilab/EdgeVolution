# read .cpp and .cc file
with open('TfLiteSketch/airway_model_tflite.cpp', 'r') as cpp_file, open ('airway_model.cc', 'r') as cc_file:
    cpp_lines = cpp_file.readlines()
    cc_lines = cc_file.readlines()

start = 14  # C-array in .cpp file starts at index 15
with open('TfLiteSketch/airway_model_tflite.cpp', 'w') as cpp_file:
    # re-write the file header first
    for i in range(0, start):
        cpp_file.write(cpp_lines[i])

    # delete whole C-array from .cpp file
    for i in range(start, len(cpp_lines)):
        cpp_file.write("")

    # write new C-array to .cpp file
    for cc_line in cc_lines:
        cc_line = cc_line.replace("unsigned int airway_model_tflite_len", "const int airway_model_tflite_len")
        cc_line = cc_line.replace("unsigned char airway_model_tflite[]", "const unsigned char airway_model_tflite[] DATA_ALIGN_ATTRIBUTE")
        cpp_file.write(cc_line)



