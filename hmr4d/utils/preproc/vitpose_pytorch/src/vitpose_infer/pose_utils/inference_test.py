from builder import build_model
import torch
from ViTPose_trt import TRTModule_ViTPose
# pose = TRTModule_ViTPose(path='pose_higher_hrnet_w32_512.engine',device='cuda:0')
pose = build_model('ViTPose_base_coco_256x192','./models/vitpose-b.pth')
pose.cuda().eval()
if pose.training:
    print('train')
else:
    print('eval')
device = torch.device("cuda")
# pose.to(device)
dummy_input = torch.randn(10, 3,256,192, dtype=torch.float).to(device)
repetitions=100
total_time = 0
starter, ender = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
with torch.no_grad():
    for rep in range(repetitions):
        # starter, ender = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
        starter.record()
        # for k in range(10):
        _ = pose(dummy_input)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)/1000
        total_time += curr_time
Throughput =   repetitions*10/total_time
print('Final Throughput:',Throughput)
print('Total time',total_time)