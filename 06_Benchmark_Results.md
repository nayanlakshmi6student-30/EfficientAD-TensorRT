# Benchmark Results

## Command

```bash
/usr/src/tensorrt/bin/trtexec \
--loadEngine=student_cable.engine
```

## Performance Summary

| Metric | Value |
|----------|----------|
| Throughput | 164.47 FPS |
| Mean Latency | 6.84 ms |
| GPU Compute Time | 6.06 ms |
| Engine Size | 8.5 MB |

## Status

```text
PASSED TensorRT.trtexec
```

## Conclusion

The EfficientAD model was successfully converted to TensorRT FP16 format and executed on Jetson Orin with real-time inference performance.