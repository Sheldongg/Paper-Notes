图像生成里面的Lora是基于Stable Diffusion 进行秩分解来减少模型训练参数的方法

```python
class LoRAModule(torch.nn.Module):
    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,  # 这是低秩分解的秩,通常远小于原始维度
        alpha=1,
        ...
    ):
        # 获取原始层的输入输出维度
        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels  # 原始输入维度
            out_dim = org_module.out_channels  # 原始输出维度
        else:
            in_dim = org_module.in_features  # 原始输入维度
            out_dim = org_module.out_features  # 原始输出维度

        self.lora_dim = lora_dim  # 低秩维度

        # 创建低秩分解的两个矩阵
        if org_module.__class__.__name__ == "Conv2d":
            # down投影: in_dim -> lora_dim (降维)
            self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
            # up投影: lora_dim -> out_dim (升维)
            self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
        else:
            # down投影: in_dim -> lora_dim (降维)
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            # up投影: lora_dim -> out_dim (升维)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)
```
lora 类几个关键定义
* lora_down - layer_up 是两个线性层，用于将原始输入和输出维度进行降维和升维
* .__class__.__name__ 来进行lora应用层的判断
