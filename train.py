"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)
# 实际流程是：
# @hydra.main装饰器会根据config_path找到配置文件目录
# 然后根据命令行参数--config-name加载指定的配置文件（如train_diffusion_lowdim_workspace.yaml）
# 最后将解析后的配置内容作为参数传递给main函数，也就是cfg参数。
# 这样，你就可以在main函数中使用cfg来访问和操作这些配置参数。
# config_path只是告诉Hydra去哪里找配置文件，而cfg参数接收的是从那些配置文件中解析出来的实际配置内容。
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    # 解析配置文件中的所有${now:}解析器，确保它们使用相同的时间。
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
