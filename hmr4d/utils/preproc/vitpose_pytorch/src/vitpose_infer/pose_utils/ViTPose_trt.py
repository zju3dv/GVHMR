import tensorrt as trt
import torch.nn
from collections import OrderedDict, namedtuple
import numpy as np

def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device("cuda")
    elif device == trt.TensorLocation.HOST:
        return torch.device("cpu")
    else:
        return TypeError("%s is not supported by torch" % device)
def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif trt.__version__ >= '7.0' and dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError("%s is not supported by torch" % dtype)
class TRTModule_ViTPose(torch.nn.Module):
    def __init__(self, engine=None, input_names=None, output_names=None, input_flattener=None, output_flattener=None,path=None,device=None):
        super(TRTModule_ViTPose, self).__init__()
        # self._register_state_dict_hook(TRTModule._on_state_dict)
        # self.engine = engine
        logger = trt.Logger(trt.Logger.INFO)
        with open(path, 'rb') as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is not None:
            self.context = self.engine.create_execution_context()
        self.input_names = ['images']
        self.output_names = []
        self.input_flattener = input_flattener
        self.output_flattener = output_flattener
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        
        # with open(path, 'rb') as f, trt.Runtime(logger) as runtime:
        #     self.model = runtime.deserialize_cuda_engine(f.read())
        # self.context = self.model.create_execution_context()
        self.bindings = OrderedDict()
        # self.output_names = []
        fp16 = False  # default updated below
        dynamic = False
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            if self.engine.binding_is_input(i):
                if -1 in tuple(self.engine.get_binding_shape(i)):  # dynamic
                    dynamic = True
                    self.context.set_binding_shape(i, tuple(self.engine.get_profile_shape(0, i)[2]))
                if dtype == np.float16:
                    fp16 = True
            else:  # output
                self.output_names.append(name)
            shape = tuple(self.context.get_binding_shape(i))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.batch_size = self.bindings['images'].shape[0] 

    

    def forward(self, *inputs):
        bindings = [None] * (len(self.input_names) + len(self.output_names))
        
        if self.input_flattener is not None:
            inputs = self.input_flattener.flatten(inputs)

        for i, input_name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(input_name)
            shape = tuple(inputs[i].shape)
            bindings[idx] = inputs[i].contiguous().data_ptr()
            self.context.set_binding_shape(idx, shape)

        # create output tensors
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = tuple(self.context.get_binding_shape(idx))
            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
            bindings[idx] = output.data_ptr()

        self.context.execute_async_v2(
            bindings, torch.cuda.current_stream().cuda_stream
        )

        if self.output_flattener is not None:
            outputs = self.output_flattener.unflatten(outputs)
        else:
            outputs = tuple(outputs)
            if len(outputs) == 1:
                outputs = outputs[0]

        return outputs