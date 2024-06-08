import torch


def quaternion_to_yaw(quat: torch.Tensor, device="cpu") -> torch.Tensor:
    single_quaternion = len(quat.shape) == 1
    
    if single_quaternion:
        quat = quat.unsqueeze(0)  # Convert single quaternion to batch with size 1
    
    no_rotation_mask = torch.all(quat == torch.tensor([0., 0., 0., 1.], device=device), dim=1)
    
    sin_yaw = 2.0 * (quat[:, -1] * quat[:, 2] + quat[:, 0] * quat[:, 1])
    cos_yaw = quat[:, -1] * quat[:, -1] + quat[:, 0] * quat[:, 0] - quat[:, 1] * quat[:, 1] - quat[:, 2] * quat[:, 2]
    yaw = torch.atan2(sin_yaw, cos_yaw)
    
    yaw[no_rotation_mask] = 0.0
    
    if single_quaternion:
        yaw = yaw.squeeze()

    return yaw