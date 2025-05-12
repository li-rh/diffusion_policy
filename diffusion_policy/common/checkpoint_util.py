from typing import Optional, Dict
import os

class TopKCheckpointManager:
    def __init__(self,
            save_dir,
            monitor_key: str,
            mode='min',
            k=1,
            format_str='epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt'
        ):
        assert mode in ['max', 'min']
        assert k >= 0

        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = k
        self.format_str = format_str
        self.path_value_map = dict()
    
    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        """
        根据监控指标值决定是否保存新的检查点，并返回检查点路径
        
        Args:
            data: 包含监控指标和其他日志数据的字典
            
        Returns:
            如果需要保存新检查点则返回路径字符串，否则返回None
        """
        # 如果k=0表示不保存任何检查点
        if self.k == 0:
            return None

        # 从输入数据中获取监控指标值
        value = data[self.monitor_key]
        # 根据格式字符串生成检查点路径
        ckpt_path = os.path.join(
            self.save_dir, self.format_str.format(**data))
        
        # 如果当前保存的检查点数量未达到k个
        if len(self.path_value_map) < self.k:
            # 直接保存新检查点
            self.path_value_map[ckpt_path] = value
            return ckpt_path
        
        # 已保存k个检查点的情况
        # 对现有检查点按指标值排序
        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])
        min_path, min_value = sorted_map[0]  # 最小值
        max_path, max_value = sorted_map[-1]  # 最大值

        # 确定是否需要替换现有检查点
        delete_path = None
        if self.mode == 'max':
            # 最大化模式：新值大于最小值时才替换
            if value > min_value:
                delete_path = min_path
        else:
            # 最小化模式：新值小于最大值时才替换
            if value < max_value:
                delete_path = max_path

        # 不需要替换的情况
        if delete_path is None:
            return None
        else:
            # 执行替换操作
            del self.path_value_map[delete_path]  # 从map中删除旧记录
            self.path_value_map[ckpt_path] = value  # 添加新记录

            # 确保保存目录存在
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

            # 删除旧的检查点文件
            if os.path.exists(delete_path):
                os.remove(delete_path)
            return ckpt_path  # 返回新检查点路径