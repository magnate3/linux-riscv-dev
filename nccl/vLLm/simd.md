

针对FNN中的激活函数（如SwiGLU, GELU, SiLU）进行SIMD优化、查表（LUT）或分段逼近（Piecewise Approximation），是提升推理性能的关键。

+ SIMD 查表法 (Table Lookup / LUT)     
+ SIMD 分段逼近法 (Piecewise Approximation)分段逼近  

##  SwiGLU SIMD 优化技术
 ggml_swiglu  