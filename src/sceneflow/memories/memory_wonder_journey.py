import torch

class WonderJourneyMemory:
    def __init__(self, device='cuda'):
        self.device = device
        
        # 核心数据存储 (Key-Value 结构)
        self.data = {
            'points_3d': torch.tensor([], device=device),
            'colors': torch.tensor([], device=device),
            # 未来可以扩展: 'trajectory': [], 'history_images': []
        }

    def get_point_cloud(self):
        """获取当前所有的点云数据"""
        return self.data['points_3d'], self.data['colors']

    def update_point_cloud(self, new_points, new_colors):
        """
        更新点云数据 (拼接)
        """
        if new_points is None or len(new_points) == 0:
            return

        # 拼接旧数据和新数据
        self.data['points_3d'] = torch.cat([self.data['points_3d'], new_points], dim=0)
        self.data['colors'] = torch.cat([self.data['colors'], new_colors], dim=0)

    def reset(self):
        """清空记忆"""
        self.data['points_3d'] = torch.tensor([], device=self.device)
        self.data['colors'] = torch.tensor([], device=self.device)

    def remove_points(self, keep_mask):
        """
        根据掩码保留点 (用于遮挡剔除)
        keep_mask: bool tensor, True 表示保留，False 表示删除
        """
        if keep_mask is None: return
        
        # 确保 mask 长度和点数一致
        if keep_mask.shape[0] != self.data['points_3d'].shape[0]:
            print("Warning: Mask shape mismatch in memory removal.")
            return

        self.data['points_3d'] = self.data['points_3d'][keep_mask]
        self.data['colors'] = self.data['colors'][keep_mask]