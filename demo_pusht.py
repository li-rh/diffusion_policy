import numpy as np
import click
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
import pygame

@click.command()
@click.option('-o', '--output', required=True)
@click.option('-rs', '--render_size', default=96, type=int)
@click.option('-hz', '--control_hz', default=10, type=int)
def main(output, render_size, control_hz):
    """
    配合pygame来收集演示数据，用于Push-T任务。
    
    参数: 
        - -o: 输出路径，用于保存演示数据。
        - -rs: 渲染图像的大小，默认为96。
        - -hz: 控制频率，默认为10。
    用法：python demo_pusht.py -o data/pusht_demo.zarr

    Collect demonstration for the Push-T task.
    
    Usage: python demo_pusht.py -o data/pusht_demo.zarr
    
    This script is compatible with both Linux and MacOS.
    Hover mouse close to the blue circle to start.
    Push the T block into the green area. 
    The episode will automatically terminate if the task is succeeded.
    Press "Q" to exit.
    Press "R" to retry.
    Hold "Space" to pause.
    """
    
    # create replay buffer in read-write mode
    replay_buffer = ReplayBuffer.create_from_path(output, mode='a')

    # create PushT env with keypoints
    kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
    env = PushTKeypointsEnv(render_size=render_size, render_action=False, **kp_kwargs) # 创建环境，返回值是一个Gym环境
    agent = env.teleop_agent() # 创建鼠标控制的agent，返回值是一个有名元组，包含act方法来获取鼠标位置
    clock = pygame.time.Clock()
    
    # episode-level while loop
    while True:
        episode = list() # 存储一个episode的数据，每个episode是一个字典，包含img, state, keypoint, action, n_contacts
        # record in seed order, starting with 0
        seed = replay_buffer.n_episodes
        print(f'starting seed {seed}')

        # set seed for env
        env.seed(seed)
        
        # reset env and get observations (including info and render for recording)
        obs = env.reset()
        info = env._get_info()
        # 如果启用了pygame的时钟，则每次调用都会返回当前画面
        img = env.render(mode='human')
        
        # loop state
        retry = False
        pause = False
        done = False
        plan_idx = 0
        pygame.display.set_caption(f'plan_idx:{plan_idx}')
        # step-level while loop
        while not done:
            # 处理按键逻辑，包括退出（Q），重试（R）进入下一次尝试，暂停（长按空格键）
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # hold Space to pause
                        plan_idx += 1
                        pygame.display.set_caption(f'plan_idx:{plan_idx}')
                        pause = True
                    elif event.key == pygame.K_r:
                        # press "R" to retry
                        retry=True
                    elif event.key == pygame.K_q:
                        # press "Q" to exit
                        exit(0)
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE:
                        pause = False

            if retry:
                break
            if pause:
                continue
            # 处理按键逻辑


            # 获取鼠标位置，如果鼠标位置与agent距离小于30 units，或者开启了telep，返回鼠标位置，否则返回None
            # 这里的obs传入参数没有被使用
            act = agent.act(obs) 
            if not act is None:
                # teleop started
                # 这里的state是agent的位置和block的位置与角度的拼接，state dim 2+3
                state = np.concatenate([info['pos_agent'], info['block_pose']])
                # discard unused information such as visibility mask and agent pos
                # for compatibility
                keypoint = obs.reshape(2,-1)[0].reshape(-1,2)[:9] 
                print(f'keypoint: {keypoint}')
                # 保存数据到episode中，每个episode是一个字典，包含img, state, keypoint, action, n_contacts
                data = {
                    'img': img,
                    'state': np.float32(state),
                    'keypoint': np.float32(keypoint),
                    'action': np.float32(act),
                    'n_contacts': np.float32([info['n_contacts']])
                }
                episode.append(data)
                
            # step env and render
            obs, reward, done, info = env.step(act)
            img = env.render(mode='human')
            
            # 设置控制频率
            clock.tick(control_hz)

        # 如果不是用户重试，说明这次演示是有效的，因此将episode保存到replay buffer中
        if not retry:
            # save episode buffer to replay buffer (on disk)
            data_dict = dict()
            for key in episode[0].keys(): # 获取episode的所有键值
                # 将episode中的所有相同键值对应的值拼接起来，组成一个数组，作为data_dict的值，
                # 例如：将所有img拼接起来，组成一个数组，作为data_dict['img']的值，方便后面使用字典的方式将所有的img获取
                data_dict[key] = np.stack(
                    [x[key] for x in episode]) 
            replay_buffer.add_episode(data_dict, compressors='disk')
            print(f'saved seed {seed}')
        else:
            print(f'retry seed {seed}')


if __name__ == "__main__":
    main()
