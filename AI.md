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
