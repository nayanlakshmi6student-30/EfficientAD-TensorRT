# TensorRT Conversion

## ONNX Model

```text
/home/pixiq/Student_cable.onnx
```

## Conversion Command

```bash
/usr/src/tensorrt/bin/trtexec \
--onnx=/home/pixiq/Student_cable.onnx \
--saveEngine=student_cable.engine \
--fp16
```

## Output

TensorRT engine generated successfully.

```text
student_cable.engine
```

## Engine Size

8.5 MB