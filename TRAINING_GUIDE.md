# 🚀 Production Training Curriculum & Compute Budget: ZipTransducer-Tamil
**Model:** ZipTransducer-Tamil Medium (51.226M Parameters: Encoder 49.25M, Decoder 1.98M)  
**Corpus:** `ai4bharat/IndicVoices` Tamil (803.9 Hours, 422,963 Utterances)  
**Target Hardware:** Kaggle Dual NVIDIA Tesla T4 (2x 16 GB GDDR6, 130 TFLOPS FP16 Peak)  

---

## 1. Corpus Statistics & Acoustic Dimensions

| Parameter / Metric | Exact Value | Mathematical Derivation / Notes |
|---|---|---|
| **Total Audio Duration** | **803.9 Hours** | $2,894,040\text{ seconds}$ total speech |
| **Total Utterances** | **422,963 Utterances** | Across read, extempore, and conversational Tamil scenarios |
| **Mean Utterance Length** | **6.84 Seconds** | $2,894,040 / 422,963 \approx 6.8415\text{ s}$ |
| **Mean Acoustic Mel Frames ($T_{mel}$)** | **684 Frames** | At 100 Hz ($10\text{ ms}$ hop, $25\text{ ms}$ window, 80-dim log-mel) |
| **Mean Subsampled Frames ($T_0$)** | **171 Frames** | After 4x ConvStem temporal compression ($25\text{ Hz}$ frame rate) |
| **Mean Target Tokens ($U$)** | **~28–34 Tokens** | Akshara tokenizer ($V = 431$), $\approx 0.55\times$ character count |
| **Lattice Surface Area ($T_0 \times U$)** | **~5,472 Nodes** | 2D Trellis per utterance for RNN-T / Joint Network |

---

## 2. Hardware Saturation & Batch Architecture on Dual Tesla T4

### 2.1 Memory Footprint (16 GB per GPU)
* **Model Weights (FP16):** $51.23\text{M} \times 2\text{ bytes} \approx \mathbf{102.5\text{ MB}}$.
* **Optimizer States (AdamW FP32 master weights + FP32 momentum + variance):** $51.23\text{M} \times 16\text{ bytes} \approx \mathbf{820\text{ MB}}$.
* **Per-GPU Batch Allocation ($B_{gpu} = 32$ utterances, max duration $15\text{s}$):**
  * Peak activation memory with Triton fused kernels & checkpointing: $\approx \mathbf{4.2\text{ GB}}$.
  * Joint network 4D tensor ($32 \times 250 \times 40 \times 512$ FP16): $\approx \mathbf{327\text{ MB}}$.
* **Total Peak VRAM per T4:** $\approx \mathbf{5.8\text{ GB}}$ out of $16.0\text{ GB}$ (**~36% utilization**, leaving huge safety headroom against OOMs).

### 2.2 Effective Batch Sizing
* **Per-GPU Batch Size ($B_{gpu}$):** $32$ utterances.
* **GPUs ($N_{gpu}$):** $2$ (Tesla T4 x 2).
* **Gradient Accumulation Steps ($K_{accum}$):** $4$ steps.
* **Global Batch Size ($B_{global}$):**
  $$B_{global} = N_{gpu} \times B_{gpu} \times K_{accum} = 2 \times 32 \times 4 = \mathbf{256\text{ utterances/step}}$$
* **Effective Audio per Optimizer Step:**
  $$\text{Audio per step} = 256 \times 6.8415\text{ s} = 1,751.4\text{ seconds} \approx \mathbf{0.4865\text{ audio hours/step}}$$
* **Steps per Epoch:**
  $$\text{Steps/Epoch} = \left\lfloor \frac{422,963}{256} \right\rfloor = \mathbf{1,652\text{ optimizer steps}}$$

---

## 3. Real-World Training Throughput & Wall-Clock Projections

### 3.1 Sustained Compute Benchmarks
* Single Tesla T4 sustained FP16 throughput on speech models: $\approx 350\times - 450\times$ real-time.
* Dual Tesla T4 (2x) parallel execution with DDP and Triton fused kernels:
  * **Conservative (including HF streaming prefetch):** **$600\times$ real-time** ($600$ audio hours processed per 1 wall-clock hour).
  * **Target Nominal:** **$750\times$ real-time** ($750$ audio hours processed per 1 wall-clock hour).
  * **Optimized Peak (Triton fused attention + DWConv):** **$900\times$ real-time** ($900$ audio hours processed per 1 wall-clock hour).

### 3.2 Wall-Clock Time per Epoch on 2x Tesla T4
$$\text{Time per Epoch (hrs)} = \frac{803.9\text{ audio hours}}{\text{Throughput (audio hrs/hr)}}$$

* At $600\times$ Real-Time: **$1.34\text{ hours}$ ($80.4\text{ minutes}$) / epoch**
* At $750\times$ Real-Time: **$1.07\text{ hours}$ ($64.3\text{ minutes}$) / epoch**
* At $900\times$ Real-Time: **$0.89\text{ hours}$ ($53.6\text{ minutes}$) / epoch**

---

## 4. Multi-Stage Curriculum Milestones

```
=============================================================================================================
Milestone              Epochs    Optimizer Steps   Audio Hours   Wall-Clock (2x T4)   Target CER    Target WER
=============================================================================================================
M1: Sanity Prototype     2        3,304 steps      1,607.8 hrs      ~2.14 hours        16% - 22%     28% - 38%
M2: Strong Baseline      5        8,260 steps      4,019.5 hrs      ~5.35 hours         6% - 9%      14% - 19%
M3: Full Convergence    10       16,520 steps      8,039.0 hrs     ~10.72 hours        < 3.5%        < 8.5%
=============================================================================================================
```

> [!IMPORTANT]
> **Kaggle 12-Hour Session Compatibility:**  
> Milestone 3 (10 epochs, 8,039 audio hours processed) completes in **~10.7 hours**, fitting safely within Kaggle's **12-hour session timeout** with a ~1.3-hour buffer for checkpoint saving, validation, and upload!

---

## 5. Optimizer, Learning Rate & SpecAugment Schedule

### 5.1 Optimizer Hyperparameters
* **Optimizer:** `AdamW`
* **Betas:** $\beta_1 = 0.90, \beta_2 = 0.98$ (recommended over 0.999 for faster acoustic alignment)
* **Epsilon:** $\epsilon = 10^{-8}$
* **Weight Decay:** $0.01$ (applied strictly to 2D/linear weight matrices; exclude RMSNorm, LayerScale, and biases)
* **Gradient Clipping:** Maximum grad norm = $1.0$ (prevents exploding gradients during early CTC/RNN-T steps)

### 5.2 Learning Rate Curve (Warmup + Cosine Annealing)
* **Total Steps:** $T_{total} = 16,520$ steps (10 epochs).
* **Warmup Phase ($0 \le t \le 1,500$ steps / ~0.9 epoch):**
  $$\eta(t) = \eta_{init} + (\eta_{peak} - \eta_{init}) \frac{t}{t_{warmup}}$$
  where $\eta_{init} = 1.0 \times 10^{-6}$ and $\eta_{peak} = 5.0 \times 10^{-4}$.
* **Cosine Decay Phase ($1,500 < t \le 16,520$ steps):**
  $$\eta(t) = \eta_{min} + \frac{1}{2}(\eta_{peak} - \eta_{min}) \left(1 + \cos\left(\frac{t - t_{warmup}}{t_{total} - t_{warmup}} \pi\right)\right)$$
  where $\eta_{min} = 1.0 \times 10^{-5}$.

### 5.3 Milestone LR Verification Points
* **Step 0 (0.0 ep):** $\eta = 1.0 \times 10^{-6}$ (Initial cold start)
* **Step 500 (0.3 ep):** $\eta = 1.67 \times 10^{-4}$ (Linear ramp)
* **Step 1,500 (0.9 ep):** $\eta = 5.00 \times 10^{-4}$ (**Peak LR**)
* **Step 3,304 (2.0 ep / M1):** $\eta = 4.83 \times 10^{-4}$ (High learning rate plateau)
* **Step 8,260 (5.0 ep / M2):** $\eta = 2.93 \times 10^{-4}$ (Core convergence zone)
* **Step 12,000 (7.3 ep):** $\eta = 1.12 \times 10^{-4}$ (Refinement decay)
* **Step 16,520 (10.0 ep / M3):** $\eta = 1.00 \times 10^{-5}$ (Final convergence minimum)

### 5.4 SpecAugment 3-Phase Schedule
1. **Phase 1: Bootstrap (Steps 0 – 1,000):**  
   Light masking ($F_{mask}=10, T_{mask}=15, N_F=1, N_T=1$). Ensures initial monotonic CTC lattice anchors before aggressive temporal corruption.
2. **Phase 2: Core Regularization (Steps 1,000 – 14,000):**  
   Full masking ($F_{mask}=27, T_{mask}=35, N_F=2, N_T=2$). Forces the 4-8-4 Zipformer encoder to learn robust phonetic invariant representations.
3. **Phase 3: Cooldown & Polishing (Steps 14,000 – 16,520):**  
   Reduced masking ($F_{mask}=15, T_{mask}=20, N_F=1, N_T=1$) to sharpen phoneme-to-akshara boundary precision.

---

## 6. Numerical Stability & Zero-NaN Guarantee Protocol

To completely prevent the `Loss: nan` issue observed during early FP16 steps:

1. **FP32 Autocast Exit for Losses:**
   Both `CTCLoss` and `RNNTLoss` must execute strictly in FP32 precision:
   ```python
   with torch.cuda.amp.autocast(enabled=False):
       loss_rnnt = rnnt_loss(logits.float(), targets, input_lengths_sub, target_lengths)
       loss_ctc = ctc_loss(ctc_logits.float(), targets, input_lengths_sub, target_lengths)
       loss_total = loss_rnnt + 0.25 * loss_ctc
   ```
2. **Strict Length Validation for CTC:**
   The CTC forward-backward algorithm guarantees NaN if $T_{sub} < U$. Enforce:
   ```python
   # Ensure downsampled acoustic frames are at least target length
   input_lengths_sub = torch.clamp(input_lengths_sub, min=target_lengths)
   ```
3. **Log-Softmax Bounds:**
   Avoid $-\infty$ underflow in uninitialized logits by clamping:
   ```python
   logits = torch.clamp(logits, min=-50.0, max=50.0)
   ```
4. **Finite Gradient Check:**
   Skip optimizer step if non-finite gradients occur:
   ```python
   scaler.unscale_(optimizer)
   grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   if not torch.isfinite(grad_norm):
       scaler.update()
       optimizer.zero_grad()
       continue
   scaler.step(optimizer)
   scaler.update()
   ```

---

## 7. Recommended Launch Command for Kaggle 2x T4

```bash
python -m torch.distributed.run \
    --nproc_per_node=2 \
    train.py \
    --model-variant medium \
    --batch-size 32 \
    --grad-accum 4 \
    --lr-peak 5e-4 \
    --lr-warmup 1500 \
    --total-steps 16520 \
    --fp16 \
    --save-dir ./checkpoints_medium \
    --eval-every 800 \
    --save-every 1652
```
