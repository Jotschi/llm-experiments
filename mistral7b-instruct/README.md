# Train

```bash
./run4Bit_v4.sh
```

# Test PEFT

```bash
./test-peft_v3.sh results/checkpoint-100/
```

# Merge PEFT

```bash
./merge-peft.sh  results/checkpoint-100/ dummy
```

# Test via Transformers inference

```bash
. venv/bin/activate
python test-inference.py models/dummy
```

# Test via VLLM

```bash
# Start vLLM
./start-vllm.sh models/dummy/

# Run inference
./vllm-test.sh dummy
```

