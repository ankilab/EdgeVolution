#!/usr/bin/env python
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import json
import os
import re
import sys
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema_fb


def BuiltinCodeToName(code):
  """Converts a builtin op code enum to a readable name."""
  for name, value in schema_fb.BuiltinOperator.__dict__.items():
    if value == code:
      return name
  return None


def NameListToString(name_list):
  """Converts a list of integers to the equivalent ASCII string."""
  if isinstance(name_list, str):
    return name_list
  else:
    result = ""
    if name_list is not None:
      for val in name_list:
        result = result + chr(int(val))
    return result


class OpCodeMapper:
  """Maps an opcode index to an op name."""

  def __init__(self, data):
    self.code_to_name = {}
    for idx, d in enumerate(data["operator_codes"]):
      self.code_to_name[idx] = BuiltinCodeToName(d["builtin_code"])
      if self.code_to_name[idx] == "CUSTOM":
        self.code_to_name[idx] = NameListToString(d["custom_code"])

  def __call__(self, x):
    if x not in self.code_to_name:
      s = "<UNKNOWN>"
    else:
      s = self.code_to_name[x]
    return "%s (%d)" % (s, x)


def CamelCaseToSnakeCase(camel_case_input):
  """Converts an identifier in CamelCase to snake_case."""
  s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel_case_input)
  return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def FlatbufferToDict(fb, preserve_as_numpy):
  """Converts a hierarchy of FB objects into a nested dict.

  We avoid transforming big parts of the flat buffer into python arrays. This
  speeds conversion from ten minutes to a few seconds on big graphs.

  Args:
    fb: a flat buffer structure. (i.e. ModelT)
    preserve_as_numpy: true if all downstream np.arrays should be preserved.
      false if all downstream np.array should become python arrays
  Returns:
    A dictionary representing the flatbuffer rather than a flatbuffer object.
  """
  if isinstance(fb, int) or isinstance(fb, float) or isinstance(fb, str):
    return fb
  elif hasattr(fb, "__dict__"):
    result = {}
    for attribute_name in dir(fb):
      attribute = fb.__getattribute__(attribute_name)
      if not callable(attribute) and attribute_name[0] != "_":
        snake_name = CamelCaseToSnakeCase(attribute_name)
        preserve = True if attribute_name == "buffers" else preserve_as_numpy
        result[snake_name] = FlatbufferToDict(attribute, preserve)
    return result
  elif isinstance(fb, np.ndarray):
    return fb if preserve_as_numpy else fb.tolist()
  elif hasattr(fb, "__len__"):
    return [FlatbufferToDict(entry, preserve_as_numpy) for entry in fb]
  else:
    return fb


def CreateDictFromFlatbuffer(buffer_data):
  model_obj = schema_fb.Model.GetRootAsModel(buffer_data, 0)
  model = schema_fb.ModelT.InitFromObj(model_obj)
  return FlatbufferToDict(model, preserve_as_numpy=False)


def get_ops(tflite_input, input_is_filepath=True):  # pylint: disable=invalid-name
  # Convert the model into a JSON flatbuffer using flatc (build if doesn't
  # exist.
  if input_is_filepath:
    if not os.path.exists(tflite_input):
      raise RuntimeError("Invalid filename %r" % tflite_input)
    if tflite_input.endswith(".tflite") or tflite_input.endswith(".bin"):
      with open(tflite_input, "rb") as file_handle:
        file_data = bytearray(file_handle.read())
      data = CreateDictFromFlatbuffer(file_data)
    elif tflite_input.endswith(".json"):
      data = json.load(open(tflite_input))
    else:
      raise RuntimeError("Input file was not .tflite or .json")
  else:
    data = CreateDictFromFlatbuffer(tflite_input)
  data["filename"] = tflite_input if input_is_filepath else (
      "Null (used model object)")  # Avoid special case

  # Update builtin code fields.
  for d in data["operator_codes"]:
    d["builtin_code"] = max(d["builtin_code"], d["deprecated_builtin_code"])

  for _, g in enumerate(data["subgraphs"]):
    opcode_mapper = OpCodeMapper(data)
    op_keys_to_display = [("opcode_index", opcode_mapper)]

    ops = []
    if g["operators"]:
      for idx, tensor in enumerate(g["operators"]):
        for h, mapper in op_keys_to_display:
          val = tensor[h] if h in tensor else None
          val = val if mapper is None else mapper(val)
          ops.append(val)
  return ops


def main(argv):
  try:
    tflite_input = argv[1]
  except IndexError:
    print("Usage: %s <input tflite> <output html>" % (argv[0]))
  else:
    ops = get_ops(tflite_input)
    with open("test/tflite_ops.json", "r") as f:
      data = json.load(f)    
    with open("test/tflite_ops.json", "w") as f:
      data.append(ops)
      json.dump(data, f)




if __name__ == "__main__":
  main(sys.argv)
