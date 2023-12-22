import onnx

input_path = "res19m.img.onnx"
output_path = "res19m.img.backbone.onnx"
input_names = ["images"]
output_names = ["304"]

onnx.utils.extract_model(input_path, output_path, input_names, output_names)
