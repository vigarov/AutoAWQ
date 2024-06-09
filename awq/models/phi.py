import tqdm
from typing import List, Tuple
from .base import BaseAWQForCausalLM
#from awq.utils.fused_utils import fuse_qkv
#from awq.modules.fused.block import PhiBlock
#from awq.modules.fused.model import PhiModel as AWQPhiModel
from transformers.models.phi.modeling_phi import (
    PhiDecoderLayer as OldPhiDecoderLayer,
    PhiForCausalLM as OldPhiForCausalLM,
)
#from awq.modules.fused.norm import FasterTransformerRMSNorm



class PhiAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "PhiDecoderLayer"
    max_seq_len_key = "max_position_embeddings"


    @staticmethod
    def fuse_layers(model: OldPhiForCausalLM):
        raise NotImplementedError

    @staticmethod
    def get_model_layers(model: OldPhiForCausalLM):
        return model.model.layers

    @staticmethod
    def get_act_for_scaling(module: OldPhiForCausalLM):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model: OldPhiForCausalLM, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)

    @staticmethod
    def get_layers_for_scaling(module: OldPhiDecoderLayer, input_feat, module_kwargs):
        layers = []

        #Attention:

        # attention input
        layers.append(
            dict(
                prev_op=module.input_layernorm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )

        # Maybe skip this, as done in llama, if performance is similar.
        layers.append(
            dict(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.o_proj],
                inp=input_feat["self_attn.o_proj"],
            )
        )

        # MLP:

        # linear 1
        layers.append(
            dict(
                prev_op=module.self_attn,
                layers=[
                    module.mlp.fc1
                ],
                inp=input_feat["mlp.fc1"],
                module2inspect=module.mlp,
            )
        )

        # linear 2
        layers.append(
            dict(
                prev_op=module.mlp.fc1,
                layers=[module.mlp.fc2],
                inp=input_feat["mlp.fc2"],
            )
        )

        return layers

