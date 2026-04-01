import abc
import torch.nn as nn


class BaseAnomalyModel(nn.Module, metaclass=abc.ABCMeta):
    """
    无监督异常检测模型的统一接口
    ==============================
    新模型只需继承此类并实现两个方法即可接入训练脚本。
    """

    @abc.abstractmethod
    def compute_loss(self, x, x_mark=None):
        """
        计算训练损失

        Args:
            x:      [B, W, F] 输入窗口
            x_mark: [B, W]    时间掩码（可选，部分模型不使用）

        Returns:
            dict: 必须包含 'loss' 键（用于 backward），
                  可选包含其他分量键用于日志打印，如 'mse', 'kld' 等
        """
        pass

    @abc.abstractmethod
    def compute_anomaly_score(self, x, x_mark=None):
        """
        计算每个窗口的异常分数

        Args:
            x:      [B, W, F]
            x_mark: [B, W]

        Returns:
            scores: [B] 每个窗口一个标量分数，越大越异常
        """
        pass