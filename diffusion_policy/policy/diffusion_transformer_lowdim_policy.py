from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator

class DiffusionTransformerLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: TransformerForDiffusion,
            noise_scheduler: DDPMScheduler,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_cond=False,
            pred_action_steps_only=False,
            # parameters passed to step
            **kwargs):
        super().__init__()
        if pred_action_steps_only:
            assert obs_as_cond

        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        """
        执行条件扩散采样过程，生成符合条件约束的轨迹
        
        Args:
            condition_data: 条件数据张量，形状与输出轨迹相同
            condition_mask: 布尔掩码，指示哪些位置需要应用条件约束
            cond: 额外的条件输入(可选)
            generator: 随机数生成器(可选)
            **kwargs: 传递给噪声调度器的额外参数
            
        Returns:
            trajectory: 生成的轨迹张量，满足condition_mask指定的条件约束，形状与condition_data相同
            当obs_as_cond=True时, B * T * Da
            可能还包含观测预测结果(当obs_as_cond=False时) B * T * Da+Do
        """
        model = self.model  # 获取扩散模型
        scheduler = self.noise_scheduler  # 获取噪声调度器

        # 初始化随机噪声轨迹
        trajectory = torch.randn(
            size=condition_data.shape,  # 保持与条件数据相同形状
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # 设置扩散步数
        scheduler.set_timesteps(self.num_inference_steps)

        # 反向扩散过程
        for t in scheduler.timesteps:
            # 1. 应用条件约束, 只会将condition_mask为True的位置替换为condition_data对应位置的值，也就是只是替换observation部分
            # 其他部分（action部分）是随机数字
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. 预测模型输出(噪声或轨迹)
            model_output = model(trajectory, t, cond)

            # 3. 计算前一步的轨迹: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # 最终确保条件约束被应用
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory 


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        根据观测数据预测动作序列
        
        Args:
            obs_dict: 包含观测数据的字典，必须包含'obs'键
                'obs': 形状为(B, T, Do)的观测张量
                
        Returns:
            包含预测结果的字典，必须包含'action'键
                'action': 预测的动作序列 (B, Ta, Da)
                'action_pred': 完整的预测轨迹 (B, T, Da)
                可能还包含观测预测结果(当obs_as_cond=False时)
        """
        # 检查输入有效性
        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # 尚未实现历史动作处理

        # 归一化观测数据
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape  # 获取batch大小和观测维度
        To = self.n_obs_steps  # 观测步数
        assert Do == self.obs_dim  # 验证观测维度
        T = self.horizon  # 预测总步数
        Da = self.action_dim  # 动作维度

        # 准备输入设备类型
        device = self.device
        dtype = self.dtype

        # 根据配置处理观测数据
        cond = None
        cond_data = None
        cond_mask = None
        # 观测作为条件输入的情况
        # cond_data: 形状为(B, T, Da)的张量，包含动作和观测
        # cond_mask: 形状为(B, T, Da)的布尔张量，指示哪些部分是条件
        # cond_data设置为全零张量，cond_mask也全为False
        if self.obs_as_cond:
            cond = nobs[:,:To]  # 取前To步观测作为条件
            shape = (B, T, Da)  # 目标形状
            # 如果仅预测动作步数
            # cond_data: 形状为(B, Ta, Da)的张量，仅包含动作
            # cond_mask: 形状为(B, Ta, Da)的布尔张量，指示哪些部分是条件
            # 设置为全零张量，mask也全为False
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)  # 仅预测动作步数的情况
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        # 如果不将观测作为条件输入，就直接将观察数据填充到cond_data中
        # 然后根据cond_mask来选择是否使用观测作为条件
        # cond_data: 形状为(B, T, Da+Do)的张量，包含动作和观测
        # cond_mask: 形状为(B, T, Da+Do)的布尔张量，指示哪些部分是条件
        # cond_data设置为全零张量，cond_mask在观察部分为True，动作部分为False
        else:
            # 观测作为输入补全的情况
            shape = (B, T, Da+Do)  # 合并动作和观测维度
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]  # 填充观测数据
            cond_mask[:,:To,Da:] = True  # 设置观测部分的mask

        # 执行条件采样
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            cond=cond,
            **self.kwargs) # (当obs_as_cond=True时, 图片作为扩散条件)shape: B * To+Ta * Da or (当obs_as_cond=False时， 观察作为扩散轨迹的一部分输入) shape: B * To+Ta * Da+Do
        
        # 反归一化预测结果
        naction_pred = nsample[...,:Da]  # 提取动作部分  shape: B * To+Ta * Da
        action_pred = self.normalizer['action'].unnormalize(naction_pred) # shape: B * To+Ta * Da

        # 获取最终动作序列
        if self.pred_action_steps_only:
            action = action_pred  # 直接使用预测结果
        else:
            start = To - 1  # 动作起始位置
            end = start + self.n_action_steps  # 动作结束位置
            action = action_pred[:,start:end]  # 截取有效动作段 shape: B * Ta * Da

        # 构建返回结果
        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not self.obs_as_cond:
            # 当观测不作为条件时，返回观测预测结果
            nobs_pred = nsample[...,Da:]  # 提取观测部分
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
            
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        """获取优化器实例
        Args:
            weight_decay: 权重衰减系数(L2正则化)
            learning_rate: 学习率
            betas: Adam优化器的beta参数(动量系数)
        Returns:
            torch.optim.Optimizer: 配置好的优化器实例
        """
        return self.model.configure_optimizers(
                weight_decay=weight_decay, 
                learning_rate=learning_rate, 
                betas=tuple(betas))

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        # handle different ways of passing observation
        cond = None
        trajectory = action
        if self.obs_as_cond:
            cond = obs[:,:self.n_obs_steps,:]
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)
        
        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
