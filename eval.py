"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint, output_dir, device):
    """主评估函数，用于加载模型并在环境中运行评估
    
    参数:
        checkpoint: 模型检查点文件路径
        output_dir: 评估结果输出目录
        device: 运行设备(cuda/cpu)
    """
    # 检查输出目录是否存在，若存在则询问是否覆盖
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 加载模型检查点
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']  # 获取配置文件
    cls = hydra.utils.get_class(cfg._target_)  # 获取工作空间类
    workspace = cls(cfg, output_dir=output_dir)  # 初始化工作空间
    workspace: BaseWorkspace  # 类型注解
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)  # 加载模型参数
    
    # 从工作空间获取策略模型
    policy = workspace.model  # 基础模型
    if cfg.training.use_ema:  # 如果使用EMA(指数移动平均)模型
        policy = workspace.ema_model
    
    # 设置运行设备并切换到评估模式
    device = torch.device(device)
    policy.to(device)
    policy.eval()  # 设置为评估模式
    
    # 运行评估，获取环境运行器
    # 这个需要好好理解，为什么要这么做？
    env_runner = hydra.utils.instantiate(  # 实例化环境运行器
        cfg.task.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(policy)  # 在环境中运行策略并获取日志
    
    # 将评估日志保存为JSON文件
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):  # 处理视频类型数据
            json_log[key] = value._path  # 只保存视频路径
        else:
            json_log[key] = value  # 保存其他类型数据
    # 写入JSON文件
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
