from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)

class TransformerForDiffusion(ModuleAttrMixin):

    def __init__(self,
            input_dim: int,       # 输入特征维度
            output_dim: int,      # 输出特征维度
            horizon: int,         # 预测时间步数
            n_obs_steps: int = None,  # 观测时间步数(可选)
            cond_dim: int = 0,    # 条件特征维度
            n_layer: int = 12,    # Transformer层数
            n_head: int = 12,     # 注意力头数
            n_emb: int = 768,     # 嵌入维度
            p_drop_emb: float = 0.1,  # 嵌入层dropout率
            p_drop_attn: float = 0.1, # 注意力层dropout率
            causal_attn: bool=False,   # 是否使用因果注意力
            time_as_cond: bool=True,   # 是否将时间作为条件
            obs_as_cond: bool=False,   # 是否将观测作为条件
            n_cond_layers: int = 0    # transform的编码器层数
        ) -> None:
        """
        Transformer模型，用于处理时间序列数据。

        Args:
            input_dim (int): 输入特征维度。
            output_dim (int): 输出特征维度。
            horizon (int): 预测时间步数。
            n_obs_steps (int, optional): 观测时间步数。默认为None。
            cond_dim (int): 条件特征维度。默认为0。
            n_layer (int): Transformer层数。默认为12。
            n_head (int): 注意力头数。默认为12。
            n_emb (int): 嵌入维度。默认为768。
            p_drop_emb (float): 嵌入层dropout率。默认为0.1。
            p_drop_attn (float): 注意力层dropout率。默认为0.1。
            causal_attn (bool): 是否使用因果注意力。默认为False。
            time_as_cond (bool): 是否将时间作为条件。默认为True。
            obs_as_cond (bool): 是否将观测作为条件。默认为False。
            n_cond_layers (int): transform的编码器层数。默认为0。
        """
        super().__init__()

        # 计算主分支和条件编码器的token数量
        if n_obs_steps is None:
            n_obs_steps = horizon
        
        T = horizon  # 主序列长度

        # 如果有条件编码，就使用encoder和decoder架构，否则使用BERT风格的encoder-only架构
        T_cond = 1    # 条件序列初始长度
        if not time_as_cond:
            T += 1     # 时间不作为条件时增加主序列长度
            T_cond -= 1
        obs_as_cond = cond_dim > 0  # 根据条件维度确定是否使用观测条件
        if obs_as_cond:
            assert time_as_cond  # 观测作为条件时需要时间也作为条件
            T_cond += n_obs_steps  # 增加观测步数到条件序列

        # 输入嵌入层
        self.input_emb = nn.Linear(input_dim, n_emb)  # 线性变换输入特征 shape: [B, T, n_emb]
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))  # 可学习位置编码 shape: [1, T, n_emb]
        self.drop = nn.Dropout(p_drop_emb)  # 嵌入层dropout

        ###################
        # 时间位置编码器  #
        ###################
        self.time_emb = SinusoidalPosEmb(n_emb)  # 时间正弦位置编码
        self.cond_obs_emb = None
        
        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)  # 观测条件嵌入层

        # 初始化编码器/解码器
        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        encoder_only = False
        
        ###################
        #  编码器配置      #
        ###################
        # 如果有条件编码的情况，则使用Transformer编码器
        # 如果没有条件编码，则使用BERT风格的encoder-only架构
        # 这里使用Transformer编码器是为了学习条件序列的特征表示
        if T_cond > 0:
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))  # 初始化条件位置编码，形状为[1, T_cond, n_emb]
            # 如果num_layers大于0，则使用Transformer编码器，否则使用简单MLP
            if n_cond_layers > 0:
                # 使用Transformer编码器, 输入和输出性质相同为 [B, T_cond, n_emb]
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4*n_emb,
                    dropout=p_drop_attn,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True
                ) 
                self.encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=n_cond_layers
                )
            else:
                # 使用简单MLP编码器
                self.encoder = nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb),
                    nn.Mish(),
                    nn.Linear(4 * n_emb, n_emb)
                )
            
            # 解码器配置
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True  # 稳定训练的关键
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=n_layer
            )
        else:
            # 仅编码器模式(BERT风格)
            encoder_only = True
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_layer
            )

        ###################
        #  注意力掩码配置  #
        ###################
        if causal_attn:
            # 因果注意力掩码(仅关注左侧序列)
            sz = T
            # 用于获取矩阵的上三角部分(包括对角线)，然后转置，作为掩码来实现因果注意力
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1) # shape: [T, T]
            # 填充mark==0的部分为-inf，以确保模型在计算softmax的时候这些位置的权重为0，以确保模型不会关注当前输入之后的信息
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            # 注册为模型的缓冲区，这样在模型训练时这些掩码会被自动加载到GPU上
            self.register_buffer("mask", mask)
            
            # 条件与观测条件同时存在时的 记忆掩码
            if time_as_cond and obs_as_cond:
                S = T_cond # 条件序列长度 等于 n_obs_steps + 1
                # 这段代码创建了两个网格坐标矩阵：
                    # t: 形状为 (T, S)，每行重复相同的值，从0到T-1
                    # s: 形状为 (T, S)，每列重复相同的值，从0到S-1
                # 这通常用于创建注意力掩码，其中t和s表示目标序列和源序列的位置索引。
                t, s = torch.meshgrid(
                    torch.arange(T),
                    torch.arange(S),
                    indexing='ij'
                )
                mask = t >= (s-1)  # 时间作为条件第一个token的特殊处理 shape: [T, S]
                mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                # 注册为模型的缓冲区，这样在模型训练时这些掩码会被自动加载到GPU上
                self.register_buffer('memory_mask', mask)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        # 输出头配置
        self.ln_f = nn.LayerNorm(n_emb)  # 最终层归一化
        self.head = nn.Linear(n_emb, output_dim)  # 输出线性层
        
        # 存储配置常量
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.encoder_only = encoder_only

        # 初始化权重
        self.apply(self._init_weights)
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self, module):
        """
        初始化模型各层的权重参数
        
        Args:
            module: 需要初始化的神经网络模块
        """
        # 定义不需要初始化的模块类型
        ignore_types = (nn.Dropout, 
            SinusoidalPosEmb, 
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential)
            
        # 线性层和嵌入层的初始化
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # 权重使用正态分布初始化(均值0,标准差0.02)
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # 如果有偏置项，初始化为0
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
        # 多头注意力层的初始化        
        elif isinstance(module, nn.MultiheadAttention):
            # 初始化所有权重矩阵
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            # 初始化所有偏置项
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
                    
        # 层归一化的初始化
        elif isinstance(module, nn.LayerNorm):
            # 偏置初始化为0，权重初始化为1
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
            
        # TransformerForDiffusion自定义模块的特殊初始化
        elif isinstance(module, TransformerForDiffusion):
            # 位置编码使用正态分布初始化
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            # 如果有条件观测嵌入层，也初始化其位置编码
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
                
        # 忽略不需要初始化的模块类型
        elif isinstance(module, ignore_types):
            pass
            
        # 未处理的模块类型报错
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        将模型参数分为两组：应用权重衰减的和不应用权重衰减的
        
        Args:
            weight_decay: 权重衰减系数，默认为1e-3
            
        Returns:
            包含两组参数的列表，每组参数都有对应的权重衰减设置
        """
        # 初始化两个集合分别存储需要和不需要权重衰减的参数名
        decay = set()
        no_decay = set()
        
        # 定义需要权重衰减的模块类型(白名单)
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        # 定义不需要权重衰减的模块类型(黑名单)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        
        # 遍历所有模块和参数
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                # 获取完整参数名(包含模块名)
                fpn = "%s.%s" % (mn, pn) if mn else pn
                
                # 处理偏置项(都不需要权重衰减)
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # 多头注意力层的偏置项命名以"bias"开头
                    no_decay.add(fpn)
                # 处理权重项(根据模块类型决定)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        
        # 特殊处理位置编码参数(不需要权重衰减)
        no_decay.add("pos_emb")
        no_decay.add("_dummy_variable")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")
        
        # 验证所有参数都已分类
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        # 确保没有参数同时出现在两个集合中
        assert len(inter_params) == 0, f"参数 {inter_params} 同时出现在两个集合中!"
        # 确保所有参数都已分类
        assert len(param_dict.keys() - union_params) == 0, f"参数 {param_dict.keys() - union_params} 未被分类!"
        
        # 创建优化器参数组
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups


    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        """
        配置模型的优化器
        
        Args:
            learning_rate: 学习率，默认为1e-4
            weight_decay: 权重衰减系数，默认为1e-3
            betas: Adam优化器的beta参数，默认为(0.9, 0.95)
            
        Returns:
            配置好的AdamW优化器实例
        """
        # 获取分组后的优化参数(应用权重衰减和不应用权重衰减的参数)
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        # 创建AdamW优化器实例
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def forward(self, 
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int], 
        cond: Optional[torch.Tensor]=None, **kwargs):
        """
        前向传播函数
        
        Args:
            sample: 输入样本张量，形状为(B,T,input_dim)
            timestep: 扩散时间步，可以是张量(B,)或标量
            cond: 可选条件输入，形状为(B,T',cond_dim)
            **kwargs: 其他关键字参数
            
        Returns:
            输出张量，形状为(B,T,output_dim)
        """
        # 1. 处理时间步输入
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # 将标量时间步转换为张量
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            # 处理0维张量情况
            timesteps = timesteps[None].to(sample.device)
        # 将时间步广播到batch维度
        timesteps = timesteps.expand(sample.shape[0])
        # 获取时间嵌入 (B,1,n_emb)
        time_emb = self.time_emb(timesteps).unsqueeze(1)

        # 2. 处理输入样本
        input_emb = self.input_emb(sample)  # (B,T,n_emb)

        if self.encoder_only:
            # BERT风格处理(仅编码器)
            # 拼接时间嵌入和输入嵌入
            token_embeddings = torch.cat([time_emb, input_emb], dim=1)
            t = token_embeddings.shape[1]
            # 获取位置嵌入
            position_embeddings = self.pos_emb[:, :t, :]
            # 应用dropout
            x = self.drop(token_embeddings + position_embeddings)
            # 通过编码器 (B,T+1,n_emb)
            x = self.encoder(src=x, mask=self.mask)
            # 移除时间token (B,T,n_emb)
            x = x[:,1:,:]
        else:
            # 编码器-解码器架构处理
            
            # 2.1 编码器部分
            cond_embeddings = time_emb
            if self.obs_as_cond:
                # 如果有观测条件，处理条件输入
                cond_obs_emb = self.cond_obs_emb(cond)
                cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)
            
            tc = cond_embeddings.shape[1]
            # 获取条件位置嵌入
            position_embeddings = self.cond_pos_emb[:, :tc, :]
            # 应用dropout并通过编码器
            x = self.drop(cond_embeddings + position_embeddings)
            x = self.encoder(x)
            memory = x  # 编码器输出作为解码器的memory
            
            # 2.2 解码器部分
            token_embeddings = input_emb
            t = token_embeddings.shape[1]
            # 获取主序列位置嵌入
            position_embeddings = self.pos_emb[:, :t, :]
            # 应用dropout
            x = self.drop(token_embeddings + position_embeddings)
            # 通过解码器 (B,T,n_emb)
            x = self.decoder(
                tgt=x,
                memory=memory,
                tgt_mask=self.mask,
                memory_mask=self.memory_mask
            )
        
        # 3. 输出头处理
        x = self.ln_f(x)  # 层归一化
        x = self.head(x)  # 线性投影到输出维度
        return x  # (B,T,n_out)


def test():
    # GPT with time embedding
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        # cond_dim=10,
        causal_attn=True,
        # time_as_cond=False,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    out = transformer(sample, timestep)
    print(out)
    print(out.shape)

    # GPT with time embedding and obs cond
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        cond_dim=10,
        causal_attn=True,
        # time_as_cond=False,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()
    
    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    cond = torch.zeros((4,4,10))
    out = transformer(sample, timestep, cond)
    print(out)
    print(out.shape)

    # GPT with time embedding and obs cond and encoder
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        cond_dim=10,
        causal_attn=True,
        # time_as_cond=False,
        n_cond_layers=4
    )
    opt = transformer.configure_optimizers()
    
    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    cond = torch.zeros((4,4,10))
    out = transformer(sample, timestep, cond)

    # BERT with time embedding token
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        # cond_dim=10,
        # causal_attn=True,
        time_as_cond=False,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    out = transformer(sample, timestep)

# import fire

# if __name__ == '__main__':
#     fire.Fire()