# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import pickle
from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor

param = {
    "conv1.w" : "backbone.stem.conv1.weight",
    "res.conv1.bn.b" : "backbone.stem.bn1.beta",
    "res.conv1.bn.s" : "backbone.stem.bn1.gamma",

    "res2" : "layer1",
    "res3" : "layer2",
    "res4" : "layer3",
    "res5" : "layer4",

    "branch2a.w" : "conv1.weight",
    "branch2a.bn.b" : "bn1.beta",
    "branch2a.bn.s" : "bn1.gamma",
    "branch2b.w" : "conv2.weight",
    "branch2b.bn.b" : "bn2.beta",
    "branch2b.bn.s" : "bn2.gamma",
    "branch2c.w" : "conv3.weight",
    "branch2c.bn.b" : "bn3.beta",
    "branch2c.bn.s" : "bn3.gamma",

    "branch1.w" : "conv_down_sample.weight",
    "branch1.bn.s" : "bn_down_sample.gamma",
    "branch1.bn.b" : "bn_down_sample.beta",
    "fc1000.b" : "end_point.bias",
    "fc1000.w" : "end_point.weight",
}
par_dict = pickle.load(open("../MSRA/R-101.pkl", 'rb'), encoding='iso-8859-1')
    
with open("convert_test.txt", "wt") as f:    
    new_params_list = []
    for name in par_dict:
        param_dict = {}
        parameter = par_dict[name]
        print("Before: "+name,file=f)
        name = name.replace('_','.')
        
        for fix in param:
            if fix in name:
                name = name.replace(fix, param[fix])
        
        if name.startswith("layer1") or name.startswith("layer2") or name.startswith("layer3"):
            param_name = "backbone." + name
        else:
            param_name = name
        print("After: "+param_name,file=f)
        param_dict['name'] = param_name
        param_dict['data'] = Tensor(parameter)
        new_params_list.append(param_dict)
    save_checkpoint(new_params_list, 'backbone.ckpt')