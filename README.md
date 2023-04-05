
### Plan
1. 基础instruction训练，显存记录
2. 加大模型，量化推理,
3. 扩展ds，验证ds有效性
4. 使用lora

### 关于基础模型
glm-6b
llama-7b
bloom-1b 7b
gpt2-0.125b 1.5b



```python
import torch
torch.cuda.memory_allocated()/1024/1024
torch.cuda.max_memory_allocated()/1024/1024
torch.cuda.memory_summary()
```

优化模型训练原理
https://huggingface.co/docs/transformers/v4.13.0/en/performance
https://zhuanlan.zhihu.com/p/608634079


模型量化基础
https://blog.csdn.net/WZZ18191171661/article/details/103332338


peft是技巧性微调
> PEFT approaches only fine-tune a small number of (extra) model parameters while freezing most parameters of the pretrained LLMs

https://huggingface.co/blog/peft

