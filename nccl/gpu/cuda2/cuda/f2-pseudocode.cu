

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <float.h>


// -----------------------------------------------------------------------------
// Fixed parameters for our example.
// These correspond to the block sizes from the paper:
//   B_r: query block (row block) size, B_c: key block (column block) size,
//   d: head dimension (here simplified to D = 1).
// -----------------------------------------------------------------------------
#define BLOCK_R 2  // B_r: Number of query rows per block (e.g., Q_i ∈ R^(B_r × d))
#define BLOCK_C 2  // B_c: Number of keys (and values) per block (e.g., K_j and V_j ∈ R^(B_c × d))
#define D 1        // d: Head dimension


// -----------------------------------------------------------------------------
// CUDA kernel implementing FlashAttention-2 forward pass with debug prints.
// This kernel corresponds to the outer loop of Algorithm 1 in the paper:
//   It processes one block Q_i (of size B_r×d) of Q per CUDA block.
// -----------------------------------------------------------------------------
__global__ void flash_attention_forward_debug(
   const float *Q,    // Q ∈ R^(N×d) stored in HBM
   const float *K,    // K ∈ R^(N×d) stored in HBM
   const float *V,    // V ∈ R^(N×d) stored in HBM
   float *O,          // Output O ∈ R^(N×d)
   float *L,          // Logsumexp L ∈ R^(N) (one per query row)
   int N,             // Total sequence length (N)
   int d)             // Head dimension (d, here d == D)
{
   // -------------------------------------------------------------------------
   // (Paper Step 1 & 2) Divide Q into T_r blocks.
   // Each CUDA block processes one query block Q_i.
   // q_block_idx ∈ [0, T_r) and q_start is the starting row index of Q_i.
   // -------------------------------------------------------------------------
   int q_block_idx = blockIdx.x;
   int q_start = q_block_idx * BLOCK_R;


   if (threadIdx.x == 0)
       printf("Block %d: q_start = %d\n", q_block_idx, q_start);


   // -------------------------------------------------------------------------
   // (Paper Step 4) Load Q_i from HBM to on-chip SRAM.
   // Each thread processes one row from Q_i.
   // global_row is the index in Q.
   // -------------------------------------------------------------------------
   int thread_row = threadIdx.x;  // Each thread in the block handles one row.
   int global_row = q_start + thread_row;
   if (global_row >= N) return;


   // Load the query vector q (a d-dimensional vector) for this row.
   float q[D];
   for (int i = 0; i < d; i++) {
       q[i] = Q[global_row * d + i];
   }
   printf("Row %d: Q = ", global_row);
   for (int i = 0; i < d; i++) {
       printf("%f ", q[i]);
   }
   printf("\n");


   // -------------------------------------------------------------------------
   // (Paper Step 5) Initialize accumulators:
   //    O^(0)_i = 0, ℓ^(0)_i = 0, m^(0)_i = -∞.
   // In our code, we use:
   //    o_accum  ≡ O^(j)_i, l_accum ≡ ℓ^(j)_i, m_accum ≡ m^(j)_i.
   // -------------------------------------------------------------------------
   float o_accum[D] = {0.0f};  // Numerator accumulator (O^(j)_i)
   float l_accum = 0.0f;       // Denom. accumulator (ℓ^(j)_i)
   float m_accum = -FLT_MAX;   // Running maximum (m^(j)_i)


   // -------------------------------------------------------------------------
   // (Paper Step 1: Also, K and V are divided into T_c blocks of size B_c×d.)
   // Determine the total number of key blocks (T_c).
   // -------------------------------------------------------------------------
   int T_c = (N + BLOCK_C - 1) / BLOCK_C;


   // -------------------------------------------------------------------------
   // (Paper Step 6) Loop over key blocks j = 1 ... T_c.
   // For each key block K_j and corresponding V_j, update accumulators.
   // -------------------------------------------------------------------------
   for (int j = 0; j < T_c; j++) {
       int k_start = j * BLOCK_C;
       printf("Row %d: Processing key block %d (k_start = %d)\n", global_row, j, k_start);


       // ---------------------------------------------------------------------
       // (Paper Step 8) Compute partial attention scores S^(j)_i = Q_i * (K_j)^T.
       // Here, for the current row q, we compute s_block[t] = q ⋅ K[global_col] for t ∈ [0, B_c).
       // ---------------------------------------------------------------------
       float s_block[BLOCK_C];
       for (int t = 0; t < BLOCK_C; t++) {
           int global_col = k_start + t;
           float dot = 0.0f;
           if (global_col < N) {
               for (int i = 0; i < d; i++) {
                   dot += q[i] * K[global_col * d + i];
               }
               s_block[t] = dot;
           } else {
               // If global_col is out-of-bound, set s to -∞.
               s_block[t] = -FLT_MAX;
           }
       }
       printf("Row %d: S_block = ", global_row);
       for (int t = 0; t < BLOCK_C; t++) {
           printf("%f ", s_block[t]);
       }
       printf("\n");


       // ---------------------------------------------------------------------
       // (Paper Step 9) Compute local maximum for this block:
       //    local_max = rowmax(S^(j)_i)
       // ---------------------------------------------------------------------
       float local_max = -FLT_MAX;
       for (int t = 0; t < BLOCK_C; t++) {
           if (s_block[t] > local_max)
               local_max = s_block[t];
       }
       printf("Row %d: local_max in block %d = %f\n", global_row, j, local_max);


       // ---------------------------------------------------------------------
       // (Paper Step 9) Update running maximum:
       //    new_m = max(m^(j-1)_i, local_max)
       // In our code, new_m is the new running maximum.
       // ---------------------------------------------------------------------
       float new_m = (m_accum > local_max) ? m_accum : local_max;
       printf("Row %d: m_accum = %f, new_m = %f\n", global_row, m_accum, new_m);


       // ---------------------------------------------------------------------
       // (Paper Step 9) Rescale previous accumulators to new maximum:
       //    scale = exp(m^(j-1)_i - new_m)
       //    ℓ^(j)_i = exp(m^(j-1)_i - new_m) * ℓ^(j-1)_i
       //    O^(j)_i = exp(m^(j-1)_i - new_m) * O^(j-1)_i
       // ---------------------------------------------------------------------
       float scale = expf(m_accum - new_m);
       l_accum *= scale;
       for (int i = 0; i < d; i++) {
           o_accum[i] *= scale;
       }
       printf("Row %d: scale factor = %f, l_accum after scaling = %f\n", global_row, scale, l_accum);
       m_accum = new_m;  // Update m_accum to new maximum.


       // ---------------------------------------------------------------------
       // (Paper Step 9) Compute "local" softmax for this key block:
       //    P̃^(j)_i = exp(S^(j)_i - m^(j)_i)
       // and update accumulators:
       //    ℓ^(j)_i += rowsum(P̃^(j)_i)
       //    O^(j)_i += P̃^(j)_i * V_j
       // ---------------------------------------------------------------------
       for (int t = 0; t < BLOCK_C; t++) {
           int global_col = k_start + t;
           float exp_val = 0.0f;
           if (global_col < N)
               exp_val = expf(s_block[t] - m_accum);
           else
               exp_val = 0.0f;
           printf("Row %d: For key col %d, exp(s - m_accum) = %f\n", global_row, global_col, exp_val);


           // Update the denominator accumulator ℓ.
           l_accum += exp_val;
           // Update the numerator accumulator O with weighted V.
           if (global_col < N) {
               for (int i = 0; i < d; i++) {
                   o_accum[i] += exp_val * V[global_col * d + i];
               }
           }
       }
       printf("Row %d: After key block %d, l_accum = %f\n", global_row, j, l_accum);
       printf("Row %d: o_accum = ", global_row);
       for (int i = 0; i < d; i++) {
           printf("%f ", o_accum[i]);
       }
       printf("\n");
   } // End of loop over key blocks (j)


   // -------------------------------------------------------------------------
   // (Paper Step 12) Finalize the output for query row:
   //    O_i = diag(ℓ^(T_c)_i)^(-1) * O^(T_c)_i.
   // Here we normalize the numerator accumulator by the denominator.
   // -------------------------------------------------------------------------
   for (int i = 0; i < d; i++) {
       O[global_row * d + i] = o_accum[i] / l_accum;
   }
   // -------------------------------------------------------------------------
   // (Paper Step 13) Compute logsumexp for this row:
   //    L_i = m^(T_c)_i + log(ℓ^(T_c)_i).
   // -------------------------------------------------------------------------
   L[global_row] = m_accum + logf(l_accum);
   printf("Row %d: Final O = ", global_row);
   for (int i = 0; i < d; i++) {
       printf("%f ", O[global_row * d + i]);
   }
   printf(" | L = %f\n", L[global_row]);
}


//
// Main function to launch the debug kernel on a small example.
// This corresponds to the overall driver that partitions Q, K, V into blocks,
// calls the kernel for each query block (i.e., processing T_r blocks), and then writes
// the outputs O and L back to HBM.
// -----------------------------------------------------------------------------
int main() {
   // For our small example:
   int N = 4;   // Sequence length
   int d = D;   // Head dimension
   size_t size_Q = N * d * sizeof(float);
   size_t size_K = N * d * sizeof(float);
   size_t size_V = N * d * sizeof(float);
   size_t size_O = N * d * sizeof(float);
   size_t size_L = N * sizeof(float);


   // Example host data:
   // Q: [1, 2, 3, 4] will be partitioned into two blocks: Q_1 = [1, 2] and Q_2 = [3, 4].
   // K: [2, 1, 0, 3] and V: [10, 20, 30, 40] will be partitioned into two key blocks:
   // K_1 = [2, 1], V_1 = [10, 20] and K_2 = [0, 3], V_2 = [30, 40].
   float h_Q[4] = {1.0f, 2.0f, 3.0f, 4.0f};
   float h_K[4] = {2.0f, 1.0f, 0.0f, 3.0f};
   float h_V[4] = {10.0f, 20.0f, 30.0f, 40.0f};
   float h_O[4];
   float h_L[4];


   // Allocate device memory.
   float *d_Q, *d_K, *d_V, *d_O, *d_L;
   cudaMalloc(&d_Q, size_Q);
   cudaMalloc(&d_K, size_K);
   cudaMalloc(&d_V, size_V);
   cudaMalloc(&d_O, size_O);
   cudaMalloc(&d_L, size_L);


   // Copy inputs from host to device.
   cudaMemcpy(d_Q, h_Q, size_Q, cudaMemcpyHostToDevice);
   cudaMemcpy(d_K, h_K, size_K, cudaMemcpyHostToDevice);
   cudaMemcpy(d_V, h_V, size_V, cudaMemcpyHostToDevice);


   // -------------------------------------------------------------------------
   // (Paper Step 1) Divide Q into T_r blocks.
   // Launch kernel: each CUDA block processes BLOCK_R query rows.
   // The grid dimension is T_r = ceil(N / BLOCK_R).
   // -------------------------------------------------------------------------
   int grid_dim = (N + BLOCK_R - 1) / BLOCK_R;  // Number of query blocks (T_r)
   int block_dim = BLOCK_R;                       // Each block has BLOCK_R threads (dimensions (BLOCK_R,1,1))
   flash_attention_forward_debug<<<grid_dim, block_dim>>>(d_Q, d_K, d_V, d_O, d_L, N, d);
   cudaDeviceSynchronize();  // Ensure kernel finishes and printf outputs are flushed.


   // Copy the results back to host.
   cudaMemcpy(h_O, d_O, size_O, cudaMemcpyDeviceToHost);
   cudaMemcpy(h_L, d_L, size_L, cudaMemcpyDeviceToHost);


   // Print final outputs.
   printf("Final Output O:\n");
   for (int i = 0; i < N; i++) {
       printf("Row %d: %f\n", i, h_O[i]);
   }
   printf("Final LogSumExp L:\n");
   for (int i = 0; i < N; i++) {
       printf("Row %d: %f\n", i, h_L[i]);
   }


   // Free device memory.
   cudaFree(d_Q);
   cudaFree(d_K);
   cudaFree(d_V);
   cudaFree(d_O);
   cudaFree(d_L);


   return 0;
}
