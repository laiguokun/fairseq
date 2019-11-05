from collections import OrderedDict
import torch
import sys
import numpy as np
tf_model_file = ('./model.npy')
tf_model = np.load(tf_model_file, allow_pickle=True).item()
def convert_to_tensor(x, idx=None):
    if idx is None:
        return torch.FloatTensor(x['weight'])
    else:
        return torch.FloatTensor(x['weight'][idx])
    
new_model = OrderedDict()
#convert embedding
new_model['encoder.embed_language'] = convert_to_tensor(tf_model['input/input/type_embedding/lookup_table'], 0)
new_model['decoder.embed_language'] = convert_to_tensor(tf_model['input/input/type_embedding/lookup_table'], 1)
#remove the mask embed
new_model['encoder.embed_tokens.weight'] = \
    convert_to_tensor(tf_model['input/input/word_embedding/lookup_table'])[:32768]
new_model['decoder.embed_tokens.weight'] = \
    convert_to_tensor(tf_model['input/input/word_embedding/lookup_table'])[:32768]
def project_param(prefix):
    for layer_id in range(int(sys.argv[2])):
        def project_attn_param(n1, n2):
            d = int(sys.argv[3])
            #qkv proj
            q = convert_to_tensor(tf_model[prefix+'/transformer/layer_{}/{}_attn/q/kernel'.format(layer_id, n1)]).view(d, -1)
            k = convert_to_tensor(tf_model[prefix+'/transformer/layer_{}/{}_attn/k/kernel'.format(layer_id, n1)]).view(d, -1)
            v = convert_to_tensor(tf_model[prefix+'/transformer/layer_{}/{}_attn/v/kernel'.format(layer_id, n1)]).view(d, -1)
            #print(q.size())
            in_proj_weight = torch.cat([q,k,v],-1).t()
            #qkv bias
            q_bias = convert_to_tensor(tf_model[prefix+'/transformer/layer_{}/{}_attn/q/bias'.format(layer_id, n1)]).view(-1)
            k_bias = convert_to_tensor(tf_model[prefix+'/transformer/layer_{}/{}_attn/k/bias'.format(layer_id, n1)]).view(-1)
            v_bias = convert_to_tensor(tf_model[prefix+'/transformer/layer_{}/{}_attn/v/bias'.format(layer_id, n1)]).view(-1)
            in_proj_bias = torch.cat([q_bias,k_bias,v_bias],0)
            new_model[prefix+'.layers.{}.{}_attn.in_proj_weight'.format(layer_id, n2)] = in_proj_weight
            new_model[prefix+'.layers.{}.{}_attn.in_proj_bias'.format(layer_id, n2)] = in_proj_bias
            #output proj
            new_model[prefix+'.layers.{}.{}_attn.out_proj.weight'.format(layer_id, n2)] = \
                convert_to_tensor(tf_model[prefix+'/transformer/layer_{}/{}_attn/o/kernel'.format(layer_id, n1)]).view(d, -1)
            new_model[prefix+'.layers.{}.{}_attn.out_proj.bias'.format(layer_id, n2)] = \
                convert_to_tensor(tf_model[prefix+'/transformer/layer_{}/{}_attn/o/bias'.format(layer_id, n1)])
            #layer_norm
            new_model[prefix+'.layers.{}.{}_attn_layer_norm.weight'.format(layer_id, n2)] = \
                convert_to_tensor(tf_model[prefix+'/transformer/layer_{}/{}_attn/LayerNorm/gamma'.format(layer_id, n1)])
            new_model[prefix+'.layers.{}.{}_attn_layer_norm.bias'.format(layer_id, n2)] = \
                convert_to_tensor(tf_model[prefix+'/transformer/layer_{}/{}_attn/LayerNorm/beta'.format(layer_id, n1)])
        
        project_attn_param('self', 'self')
        if prefix=='decoder':
            project_attn_param('cross', 'encoder')
        new_model[prefix+'.layers.{}.final_layer_norm.weight'.format(layer_id)] = \
            convert_to_tensor(tf_model[prefix+'/transformer/layer_{}/ff/LayerNorm/gamma'.format(layer_id)])
        new_model[prefix+'.layers.{}.final_layer_norm.bias'.format(layer_id)] = \
            convert_to_tensor(tf_model[prefix+'/transformer/layer_{}/ff/LayerNorm/beta'.format(layer_id)])
        #ffn layer
        for i in range(1,3):
            new_model[prefix+'.layers.{}.fc{}.weight'.format(layer_id, i)] = \
                convert_to_tensor(tf_model[prefix+'/transformer/layer_{}/ff/layer_{}/kernel'.format(layer_id, i)]).t()
            new_model[prefix+'.layers.{}.fc{}.bias'.format(layer_id, i)] = \
                convert_to_tensor(tf_model[prefix+'/transformer/layer_{}/ff/layer_{}/bias'.format(layer_id, i)])
project_param('encoder')
project_param('decoder')
if 'model/lm_loss/bias' in tf_model:
  new_model['decoder.softmax_bias'] = convert_to_tensor(tf_model['model/lm_loss/bias'])[:-1]
torch.save(new_model, sys.argv[1])