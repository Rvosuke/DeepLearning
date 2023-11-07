import torch


def device_setting(device_type='cuda', device_id=0, num_cores=8):
    """设备设置：CPU/GPU、单卡/多卡、多线程。"""
    # 设置使用的设备
    device = torch.device(f"{device_type}:{device_id}" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(device_id)  # 设置使用的 GPU
        torch.backends.cudnn.benchmark = True  # 自动寻找最快的卷积算法
    # 设置多线程
    if device.type == 'cpu':
        torch.set_num_threads(num_cores)

    return device
