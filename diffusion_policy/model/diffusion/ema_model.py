import copy
import torch
from torch.nn.modules.batchnorm import _BatchNorm

class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(
        self,
        model,  # 原始模型，用于初始化EMA模型
        update_after_step=0,  # 开始EMA更新的训练步数
        inv_gamma=1.0,  # EMA热启动的逆乘因子
        power=2 / 3,  # EMA热启动的指数因子
        min_value=0.0,  # EMA衰减率的最小值
        max_value=0.9999  # EMA衰减率的最大值
    ):
        """
        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).
        Args:
            inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
            power (float): Exponential factor of EMA warmup. Default: 2/3.
            min_value (float): The minimum EMA decay rate. Default: 0.
        """

        # 初始化EMA模型
        self.averaged_model = model  # 保存原始模型的副本
        self.averaged_model.eval()  # 设置为评估模式
        self.averaged_model.requires_grad_(False)  # 禁用梯度计算

        # 设置EMA参数
        self.update_after_step = update_after_step  # 开始EMA更新的步数阈值
        self.inv_gamma = inv_gamma  # 控制EMA热启动速度的参数
        self.power = power  # 控制EMA热启动曲线的参数
        self.min_value = min_value  # 衰减率下限
        self.max_value = max_value  # 衰减率上限

        # 初始化训练状态
        self.decay = 0.0  # 当前衰减率
        self.optimization_step = 0  # 当前优化步数计数器

    def get_decay(self, optimization_step):
        """
        计算指数移动平均的衰减因子
         
        Args:
            optimization_step (int): 当前优化步数
             
        Returns:
            float: 计算得到的衰减率，范围在[min_value, max_value]之间
        """
        # 计算有效步数（减去延迟更新步数）
        step = max(0, optimization_step - self.update_after_step - 1)
         
        # 使用热启动公式计算衰减率
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
 
        # 如果步数不足，返回0衰减率
        if step <= 0:
            return 0.0
 
        # 确保衰减率在[min_value, max_value]范围内
        return max(self.min_value, min(value, self.max_value))
        
    @torch.no_grad()
    def step(self, new_model):
        """执行一步EMA更新
        
        Args:
            new_model: 当前训练的最新模型，用于更新EMA模型权重
        """
        # 计算当前步数的衰减率
        self.decay = self.get_decay(self.optimization_step)

        # 遍历模型的所有模块和参数
        for module, ema_module in zip(new_model.modules(), self.averaged_model.modules()):            
            for param, ema_param in zip(module.parameters(recurse=False), ema_module.parameters(recurse=False)):
                # 仅处理直接参数，不递归处理子模块
                if isinstance(param, dict):
                    raise RuntimeError('Dict parameter not supported')
                
                # 处理BatchNorm层参数
                if isinstance(module, _BatchNorm):
                    # 直接复制BatchNorm参数，不应用EMA
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                elif not param.requires_grad:
                    # 对于不需要梯度的参数，直接复制
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                else:
                    # 应用EMA更新公式: ema_param = decay*ema_param + (1-decay)*param
                    ema_param.mul_(self.decay)
                    ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1 - self.decay)

        # 更新优化步数计数器
        self.optimization_step += 1
