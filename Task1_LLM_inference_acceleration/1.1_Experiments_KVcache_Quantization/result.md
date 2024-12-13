# 实验结果
## Naive mode: no KV Cache & no quantization.
- 模型本身占用空间487.46875 MB
- Model Memory Usage: 487.46875 MB
- Inference Time (Naïve): 11.3495 seconds
- Throughput (Naïve): 87.2282 tokens/second
- Inference time GPU Memory Usage (Naïve): 1072.46 MB

## KV-cache mode: with KV Cache & no quantization.
使用KV Cache之后，推理速度大幅提升，内存峰值占用也变少。\
这是因为不用KV Cache的时候，峰值显存占用会因为重新计算每个token的KV而变得更高。

- Model Memory Usage: 487.46875 MB
- Inference Time (KV-cache): 3.8137 seconds
- Throughput (KV-cache): 259.5926 tokens/second
- GPU Memory Usage (KV-cache): 638.07 MB

## FP16 mode: with KV Cache & with FP16 quantization.
使用FP16量化，模型大小大致减半，显存占用变少。推理速度略微变慢，可能来自量化过程本身的运算量，以及优化上的一些不足。

- Model Memory Usage: 255.4873046875 MB
- Inference Time (FP16): 4.3595 seconds
- Throughput (FP16): 227.0883 tokens/second
- GPU Memory Usage (FP16): 263.62 MB

## INT8 mode: with KV Cache & with INT8 quantization.
使用INT8量化，模型大小相比FP16再次减半。但是推理速度显著变慢，与助教讨论后，可以认为是相比其他位数的量化，对INT8权重的运算优化不足。

- Model Memory Usage: 174.9912109375 MB
- Inference Time (INT8): 21.5527 seconds
- Throughput (INT8): 45.9340 tokens/second
- GPU Memory Usage (INT8): 183.82 MB

## FP4 mode: with KV Cache & with FP4 quantization.
接下来，我尝试FP4量化，并实验了文档中提到的一些高级用例。

- Model Memory Usage: 134.1982421875 MB
- Inference Time (FP4): 8.6416 seconds
- Throughput (FP4): 114.5623 tokens/second
- GPU Memory Usage (FP4): 142.33 MB

## FP4-BF16 mode: with KV Cache & with FP4-BF16 quantization.
文档中表示,计算数据类型默认float32，可以通过bnb_4bit_compute_dtype=torch.bfloat16加快速度 \
实验发现不影响内存占用，计算速度有较小提升

- Model Memory Usage: 134.1982421875 MB
- Inference Time (FP4-BF16): 8.4785 seconds
- Throughput (FP4-BF16): 116.7662 tokens/second
- GPU Memory Usage (FP4-BF16): 142.33 MB

## FP4-Double-Quant mode: with KV Cache & with FP4 quantization.
使用嵌套量化，略微减少内存占用，但是推理速度变慢

- Model Memory Usage: 130.5078125 MB
- Inference Time (FP4-Double-Quant): 9.4089 seconds
- Throughput (FP4-Double-Quant): 105.2198 tokens/second
- GPU Memory Usage (FP4-Double-Quant): 138.59 MB

# How quantization impacts the trade-off between memory and speed. 

Quantization通过减少表示数字需要的位数,减少需要的存储空间,减少传递数据需要的时间. 量化的过程可能造成额外的一部分运算,导致速度有所损失.原先KV Cache通过空间换取时间,而Quantization降低了对显存的占用和数据传输的时间,通过牺牲一部分精度,缓解了空间和速度的权衡,使得模型整体的表现得到提高.