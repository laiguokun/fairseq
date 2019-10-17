from collections import OrderedDict
import torch
import sys
def convert_to_tensor(x, idx=None):
    if idx is None:
        return torch.FloatTensor(x['weight'])
    else:
        return torch.FloatTensor(x['weight'][idx])
      
import numpy as np
tf_model_file = ('./model.npy')
tf_model = np.load(tf_model_file, allow_pickle=True).item()

new_model = OrderedDict()
#convert embedding
new_model['encoder.embed_language'] = convert_to_tensor(tf_model['model/input/type_embedding/lookup_table'], 0)
new_model['decoder.embed_language'] = convert_to_tensor(tf_model['model/input/type_embedding/lookup_table'], 1)
#remove the mask embed
new_model['encoder.embed_tokens.weight'] = \
    convert_to_tensor(tf_model['model/input/word_embedding/lookup_table'])[:-1]
new_model['decoder.embed_tokens.weight'] = \
    convert_to_tensor(tf_model['model/input/word_embedding/lookup_table'])[:-1]

for layer_id in range(14):
    #qkv proj
    q = convert_to_tensor(tf_model['model/transformer/layer_{}/abs_attn/q/kernel'.format(layer_id)]).view(1024, -1)
    k = convert_to_tensor(tf_model['model/transformer/layer_{}/abs_attn/k/kernel'.format(layer_id)]).view(1024, -1)
    v = convert_to_tensor(tf_model['model/transformer/layer_{}/abs_attn/v/kernel'.format(layer_id)]).view(1024, -1)
    #print(q.size())
    in_proj_weight = torch.cat([q,k,v],-1).t()
    #qkv bias
    q_bias = convert_to_tensor(tf_model['model/transformer/layer_{}/abs_attn/q/bias'.format(layer_id)]).view(-1)
    k_bias = convert_to_tensor(tf_model['model/transformer/layer_{}/abs_attn/k/bias'.format(layer_id)]).view(-1)
    v_bias = convert_to_tensor(tf_model['model/transformer/layer_{}/abs_attn/v/bias'.format(layer_id)]).view(-1)
    in_proj_bias = torch.cat([q_bias,k_bias,v_bias],0)
    new_model['decoder.layers.{}.self_attn.in_proj_weight'.format(layer_id)] = in_proj_weight
    new_model['decoder.layers.{}.self_attn.in_proj_bias'.format(layer_id)] = in_proj_bias
    #output proj
    new_model['decoder.layers.{}.self_attn.out_proj.weight'.format(layer_id)] = \
        convert_to_tensor(tf_model['model/transformer/layer_{}/abs_attn/o/kernel'.format(layer_id)]).view(1024, -1)
    new_model['decoder.layers.{}.self_attn.out_proj.bias'.format(layer_id)] = \
        convert_to_tensor(tf_model['model/transformer/layer_{}/abs_attn/o/bias'.format(layer_id)])
    #layer_norm
    new_model['decoder.layers.{}.self_attn_layer_norm.weight'.format(layer_id)] = \
        convert_to_tensor(tf_model['model/transformer/layer_{}/abs_attn/LayerNorm/gamma'.format(layer_id)])
    new_model['decoder.layers.{}.self_attn_layer_norm.bias'.format(layer_id)] = \
        convert_to_tensor(tf_model['model/transformer/layer_{}/abs_attn/LayerNorm/beta'.format(layer_id)])
    new_model['decoder.layers.{}.final_layer_norm.weight'.format(layer_id)] = \
        convert_to_tensor(tf_model['model/transformer/layer_{}/ff/LayerNorm/gamma'.format(layer_id)])
    new_model['decoder.layers.{}.final_layer_norm.bias'.format(layer_id)] = \
        convert_to_tensor(tf_model['model/transformer/layer_{}/ff/LayerNorm/beta'.format(layer_id)])
    #ffn layer
    for i in range(1,3):
        new_model['decoder.layers.{}.fc{}.weight'.format(layer_id, i)] = \
            convert_to_tensor(tf_model['model/transformer/layer_{}/ff/layer_{}/kernel'.format(layer_id, i)]).t()
        new_model['decoder.layers.{}.fc{}.bias'.format(layer_id, i)] = \
            convert_to_tensor(tf_model['model/transformer/layer_{}/ff/layer_{}/bias'.format(layer_id, i)])
if 'model/lm_loss/bias' in tf_model:
  new_model['decoder.softmax_bias'] = convert_to_tensor(tf_model['model/lm_loss/bias'])[:-1]
new_model['encoder.version'] = torch.FloatTensor([2.])
new_model['decoder.version'] = torch.FloatTensor([2.])
new_model['decoder.embed_positions._float_tensor'] = torch.FloatTensor([0.])
new_model['encoder.embed_positions._float_tensor'] = torch.FloatTensor([0.])
torch.save(new_model, sys.argv[1])