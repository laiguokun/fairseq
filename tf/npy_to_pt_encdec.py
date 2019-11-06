from collections import OrderedDict
import torch
import sys
import numpy as np

from npy_to_pt import embed_param, project_param, convert_to_tensor

if __name__ == '__main__':
    tf_model_file = ('./model.npy')
    tf_model = np.load(tf_model_file, allow_pickle=True).item()
       
    new_model = OrderedDict()
    
    embed_param(new_model, tf_model, 'input')
    project_param(new_model, tf_model, 'encoder')
    project_param(new_model, tf_model, 'decoder')
    
    if 'decoder/lm_loss/bias' in tf_model:
        print("Use softmax bias")
        new_model['decoder.softmax_bias'] = convert_to_tensor(
            tf_model['decoder/lm_loss/bias'])[:-2]
    
    torch.save(new_model, sys.argv[1])
