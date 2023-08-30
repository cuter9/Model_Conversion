import sys
import onnx
filename = "new.onnx"
model = onnx.load(filename)
onnx.checker.check_model(model)