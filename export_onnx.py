import torch


model=torch.load(
        "./output/1/trainings/mvtec_ad/cable/student_final.pth",
        map_location="cpu"
    )

model.eval()
dummy_input=torch.randn(1,3,256,256)
torch.onnx.export(
    model,
    dummy_input,
    "Student_cable.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)
print("ONNX model exported Successfully!")
