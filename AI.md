# AI

> CUDA_HOME does not exist, unable to compile CUDA op(s)

Solution:

```shell
conda install -c nvidia cuda-nvcc=${CUDA_VERSION}
```

Ref:

- <https://github.com/microsoft/DeepSpeed/issues/2772>
- <https://stackoverflow.com/questions/52731782/get-cuda-home-environment-path-pytorch>

---

> Model was not initialized with `Zero-3` despite being configured for DeepSpeed Zero-3. Please re-initialize your model via `Model.from_pretrained(...)` or `Model.from_config(...)` after creating your `TrainingArguments`

Solution:

```python
def model_init():
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=quantization_config)
    return model

cfg = SFTConfig(**kwargs)

trainer = SFTTrainer(
    args=cfg,
    tokenizer=tokenizer,
    model_init=model_init,
    train_dataset=datasets["train"],
    eval_dataset=datasets["test"],
    formatting_func=formatting_func,
)
```

Ref:

- <https://github.com/huggingface/transformers/issues/32901>  

---

> vLLM: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method.

Solution:

```shell
VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0,1 vllm serve ...
```

Ref:

- <https://github.com/vllm-project/vllm/issues/6152>  

---

> Unable to JIT load the async_io op due to it not being compatible due to hardware/software issue.

Solution:

```shell
apt install libaio-dev
```

---

> cannot find -lcurand

Solution:

```shell
apt install nvidia-cuda-toolkit
```

---

> Qwen: AttributeError: 'Qwen2Tokenizer' object has no attribute 'eod_id'

Ref:

- <https://github.com/QwenLM/Qwen2.5/issues/48>  

---

> FileExistsError: [Errno 17] File exists: '/opt/dlami/nvme/fft-8b/checkpoint-9/global_step9/offloaded_tensors'

Ref:

- <https://github.com/axolotl-ai-cloud/axolotl/issues/1617>  

---

> k8s vLLM: Failed to infer device type

Solution:

```yaml
spec:
  runtimeClassName: nvidia
```

Ref:

- <https://github.com/vllm-project/vllm/issues/8885>  

---

> k8s vLLM: The number of required GPUs exceeds the total number of available GPUs in the placement group

Solution:

`tensor-parallel-size` should not be bigger than `nvidia.com/gpu` limit from `resources`.  

Ref:

- <https://github.com/vllm-project/vllm/issues/3416>  

---

> Error 804: forward compatibility was attempted on non supported HW

Solution:

```shell
# keep the cuda version and nvidia version equal
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.{your_actual_version} /usr/lib/x86_64-linux-gnu/libcuda.so.1

# or:
# upgrade/downgrade related driver, and then reboot
```

Ref:

- <https://github.com/pytorch/pytorch/issues/40671>  
- <https://zhuanlan.zhihu.com/p/361545761>  
