# EfficientAD TensorRT Deployment on Jetson Orin

## Project Objective

Deploy and optimize the EfficientAD anomaly detection model using NVIDIA TensorRT on Jetson Orin for high-performance edge inference.

## Workflow

1. Dataset Preparation (MVTec AD)
2. EfficientAD Training
3. ONNX Export
4. TensorRT Conversion
5. Benchmarking
6. Documentation

## Dataset

- MVTec AD
- Classes Used:
  - Cable
  - Hazelnut

## Hardware

- NVIDIA Jetson Orin

## Frameworks

- PyTorch
- ONNX
- TensorRT

## Results

| Metric | Value |
|----------|----------|
| Throughput | 164.47 FPS |
| Mean Latency | 6.84 ms |
| GPU Compute Time | 6.06 ms |
| Engine Size | 8.5 MB |

## Status

 Deployment Completed

## Drive Link for Model

https://drive.google.com/drive/folders/1hBmay0gJDbirC1sFXNbkrl2YTb-V-X8h?usp=drive_link

## Model Performance Evaluation Report

The evaluation_report.txt is a threshold-optimized performance report that translates deep learning probabilities into actionable metrics—Accuracy, Precision, F1-Score, and Confusion Matrices—to validate industrial defect-detection readiness.
To find the Model Performance Evaluation report please go through evaluation_report.txt file.
