#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright 2025 The HuggingFace Team. All rights reserved.
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Optional, Tuple, Union
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.functional.op as ttsimF
import ttsim.front.functional.sim_nn as SimNN

from .attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    Attention,
    AttentionProcessor,
)
from .embeddings import (
    TimestepEmbedding,
    Timesteps,
)
from .unet_2d_blocks import (
    get_down_block,
    get_mid_block,
    get_up_block,
)


class UNet2DConditionModel(SimNN.Module):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["BasicTransformerBlock", "ResnetBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"]
    _skip_layerwise_casting_patterns = ["norm"]
    _repeated_blocks = ["BasicTransformerBlock"]

    def __init__(
        self,
        objname: str,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (    # type: ignore[assignment]
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        up_block_types: Tuple[str] = ("CrossAttnUpBlock2D", "UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),   # type: ignore[assignment]
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),    # type: ignore[assignment]
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        dropout: float = 0.0,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: float = 1.0,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        time_embedding_act_fn: Optional[str] = None,
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: Optional[int] = None,
        attention_type: str = "default",
        class_embeddings_concat: bool = False,
        mid_block_only_cross_attention: Optional[bool] = None,
        cross_attention_norm: Optional[str] = None,
        addition_embed_type_num_heads: int = 64,
    ):
        super().__init__()

        self.name = objname
        self.sample_size = sample_size
        self.center_input_sample = center_input_sample
        self.addition_embed_type = addition_embed_type


        if num_attention_heads is not None:
            raise ValueError(
                "At the moment it is not possible to define the number of attention heads via `num_attention_heads` because of a naming issue as described in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131. Passing `num_attention_heads` will only be supported in diffusers v0.19."
            )

        num_attention_heads = num_attention_heads or attention_head_dim

        # Check inputs
        self._check_config(
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            only_cross_attention=only_cross_attention,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            cross_attention_dim=cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            reverse_transformer_layers_per_block=reverse_transformer_layers_per_block,  # type: ignore[arg-type]
            attention_head_dim=attention_head_dim,  # type: ignore[arg-type]
            num_attention_heads=num_attention_heads,
        )

        # input
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = ttsimF.Conv2d(f'{self.name}.conv_in',
            in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        )

        # time
        time_embed_dim, timestep_input_dim = self._set_time_proj(
            time_embedding_type,
            block_out_channels=block_out_channels,  # type: ignore[arg-type]
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            time_embedding_dim=time_embedding_dim,  # type: ignore[arg-type]
        )

        self.time_embedding = TimestepEmbedding(
            f'{self.name}_TimestepEmbedding',
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
            post_act_fn=timestep_post_act,
            cond_proj_dim=time_cond_proj_dim,
        )

        self._set_encoder_hid_proj(
            encoder_hid_dim_type,
            cross_attention_dim=cross_attention_dim,
            encoder_hid_dim=encoder_hid_dim,
        )

        # class embedding
        self._set_class_embedding(
            class_embed_type,
            act_fn=act_fn,
            num_class_embeds=num_class_embeds,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            time_embed_dim=time_embed_dim,
            timestep_input_dim=timestep_input_dim,
        )

        self._set_add_embedding(
            addition_embed_type,    # type: ignore[arg-type]
            addition_embed_type_num_heads=addition_embed_type_num_heads,
            addition_time_embed_dim=addition_time_embed_dim,
            cross_attention_dim=cross_attention_dim,    # type: ignore[arg-type]
            encoder_hid_dim=encoder_hid_dim,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            time_embed_dim=time_embed_dim,
        )

        if time_embedding_act_fn is None:
            self.time_embed_act = None
        else:
            self.time_embed_act = None #get_activation(time_embedding_act_fn)

        self.down_blocks = None #nn.ModuleList([])
        self.up_blocks = None #nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            if mid_block_only_cross_attention is None:
                mid_block_only_cross_attention = only_cross_attention

            only_cross_attention = [only_cross_attention] * len(down_block_types)   # type: ignore[assignment]

        if mid_block_only_cross_attention is None:
            mid_block_only_cross_attention = False

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)    # type: ignore[assignment]

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)  # type: ignore[assignment]

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)    # type: ignore[assignment]

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)   # type: ignore[assignment]

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)   # type: ignore[assignment]

        if class_embeddings_concat:
            blocks_time_embed_dim = time_embed_dim * 2
        else:
            blocks_time_embed_dim = time_embed_dim

        # down
        output_channel = block_out_channels[0]
        down_block = [0] * len(down_block_types)
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block[i] = get_down_block(
                down_block_type,
                i,
                num_layers=layers_per_block[i], # type: ignore[index]
                transformer_layers_per_block=transformer_layers_per_block[i],   # type: ignore
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim[i], # type: ignore[index]
                num_attention_heads=num_attention_heads[i], # type: ignore[index]
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],   # type: ignore[index]
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_type=attention_type,
                resnet_skip_time_act=resnet_skip_time_act,
                resnet_out_scale_factor=resnet_out_scale_factor,
                cross_attention_norm=cross_attention_norm,
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,  # type: ignore[index]
                dropout=dropout,
            )
        self.down_blocks = SimNN.ModuleList(down_block)

        # mid
        self.mid_block = get_mid_block(
            mid_block_type, # type: ignore[arg-type]
            self.name,
            temb_channels=blocks_time_embed_dim,
            in_channels=block_out_channels[-1],
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,  # type: ignore[arg-type]
            output_scale_factor=mid_block_scale_factor,
            transformer_layers_per_block=transformer_layers_per_block[-1],  # type: ignore
            num_attention_heads=num_attention_heads[-1],    # type: ignore[index]
            cross_attention_dim=cross_attention_dim[-1],    # type: ignore[index]
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            mid_block_only_cross_attention=mid_block_only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_type=attention_type,
            resnet_skip_time_act=resnet_skip_time_act,
            cross_attention_norm=cross_attention_norm,
            attention_head_dim=attention_head_dim[-1],  # type: ignore[index]
            dropout=dropout,
        )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))  # type: ignore[arg-type]
        reversed_layers_per_block = list(reversed(layers_per_block))    # type: ignore[arg-type]
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))  # type: ignore[arg-type]
        reversed_transformer_layers_per_block = (
            list(reversed(transformer_layers_per_block))    # type: ignore[arg-type]
            if reverse_transformer_layers_per_block is None
            else reverse_transformer_layers_per_block
        )
        only_cross_attention = list(reversed(only_cross_attention)) # type: ignore

        up_block = [0] * len(up_block_types)
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block[i] = get_up_block(     # type: ignore
                up_block_type,
                i,
                self.name,
                num_layers=reversed_layers_per_block[i] + 1,
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],  # type: ignore[arg-type]
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=blocks_time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resolution_idx=i,
                resnet_groups=norm_num_groups,
                cross_attention_dim=reversed_cross_attention_dim[i],
                num_attention_heads=reversed_num_attention_heads[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],   # type: ignore[index]
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_type=attention_type,
                resnet_skip_time_act=resnet_skip_time_act,
                resnet_out_scale_factor=resnet_out_scale_factor,
                cross_attention_norm=cross_attention_norm,
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,  # type: ignore[index]
                dropout=dropout,
            )
        self.up_blocks = SimNN.ModuleList(up_block)

        # out
        if norm_num_groups is not None:
            self.conv_norm_out = SimNN.GroupNorm(f'{self.name}.conv_norm_out',
                num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
            )

            self.conv_act = SimNN.Silu(f'{self.name}.siluop') #get_activation(act_fn)

        else:
            self.conv_norm_out = None   # type: ignore[assignment]
            self.conv_act = None    # type: ignore[assignment]

        conv_out_padding = (conv_out_kernel - 1) // 2
        self.conv_out = ttsimF.Conv2d(f'{self.name}.conv_out',
            block_out_channels[0], out_channels, kernel_size=conv_out_kernel, padding=conv_out_padding
        )

        self._set_pos_net_if_use_gligen(attention_type=attention_type, cross_attention_dim=cross_attention_dim) # type: ignore[arg-type]
        super().link_op2module()

    def _check_config(
        self,
        down_block_types: Tuple[str],
        up_block_types: Tuple[str],
        only_cross_attention: Union[bool, Tuple[bool]],
        block_out_channels: Tuple[int],
        layers_per_block: Union[int, Tuple[int]],
        cross_attention_dim: Union[int, Tuple[int]],
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple[int]]],
        reverse_transformer_layers_per_block: bool,
        attention_head_dim: int,
        num_attention_heads: Optional[Union[int, Tuple[int]]],
    ):
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(only_cross_attention, bool) and len(only_cross_attention) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `only_cross_attention` as `down_block_types`. `only_cross_attention`: {only_cross_attention}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):  # type: ignore[arg-type]
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )

    def _set_time_proj(
        self,
        time_embedding_type: str,
        block_out_channels: int,
        flip_sin_to_cos: bool,
        freq_shift: float,
        time_embedding_dim: int,
    ) -> Tuple[int, int]:
        if time_embedding_type == "fourier":
            raise NotImplementedError('Fourier time embedding (time_embedding_type="fourier") is not implemented')
        elif time_embedding_type == "positional":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 4    # type: ignore[index]

            self.time_proj = Timesteps('time_proj', block_out_channels[0], flip_sin_to_cos, freq_shift) # type: ignore[index]
            timestep_input_dim = block_out_channels[0]  # type: ignore[index]
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Please make sure to use one of `fourier` or `positional`."
            )

        return time_embed_dim, timestep_input_dim

    def _set_encoder_hid_proj(
        self,
        encoder_hid_dim_type: Optional[str],
        cross_attention_dim: Union[int, Tuple[int]],
        encoder_hid_dim: Optional[int],
    ):
        if encoder_hid_dim_type is None and encoder_hid_dim is not None:
            encoder_hid_dim_type = "text_proj"
            # self.register_to_config(encoder_hid_dim_type=encoder_hid_dim_type)

        if encoder_hid_dim_type == "text_proj":
            raise NotImplementedError('Text projection encoder hidden dimension (encoder_hid_dim_type="text_proj") is not implemented')
        elif encoder_hid_dim_type == "text_image_proj":
            raise NotImplementedError('Text-image projection encoder hidden dimension (encoder_hid_dim_type="text_image_proj") is not implemented')
        elif encoder_hid_dim_type == "image_proj":
            raise NotImplementedError('Image projection encoder hidden dimension (encoder_hid_dim_type="image_proj") is not implemented')
        elif encoder_hid_dim_type is not None:
            raise ValueError(
                f"`encoder_hid_dim_type`: {encoder_hid_dim_type} must be None, 'text_proj', 'text_image_proj', or 'image_proj'."
            )
        else:
            self.encoder_hid_proj = None

    def _set_class_embedding(
        self,
        class_embed_type: Optional[str],
        act_fn: str,
        num_class_embeds: Optional[int],
        projection_class_embeddings_input_dim: Optional[int],
        time_embed_dim: int,
        timestep_input_dim: int,
    ):
        if class_embed_type is None and num_class_embeds is not None:
            raise NotImplementedError('Class embedding with num_class_embeds specified but class_embed_type=None is not implemented')
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding('TimeStepEmbedding', timestep_input_dim, time_embed_dim, act_fn=act_fn)
        elif class_embed_type == "identity":
            raise NotImplementedError('Identity class embedding (class_embed_type="identity") is not implemented')
        elif class_embed_type == "projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'projection' requires `projection_class_embeddings_input_dim` be set"
                )
            self.class_embedding = TimestepEmbedding('TimeStepEmbedding', projection_class_embeddings_input_dim, time_embed_dim)
        elif class_embed_type == "simple_projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'simple_projection' requires `projection_class_embeddings_input_dim` be set"
                )
            raise NotImplementedError('simple_projection case not implemented')
        else:
            self.class_embedding = None # type: ignore[assignment]

    def _set_add_embedding(
        self,
        addition_embed_type: str,
        addition_embed_type_num_heads: int,
        addition_time_embed_dim: Optional[int],
        flip_sin_to_cos: bool,
        freq_shift: float,
        cross_attention_dim: Optional[int],
        encoder_hid_dim: Optional[int],
        projection_class_embeddings_input_dim: Optional[int],
        time_embed_dim: int,
    ):
        if addition_embed_type == "text":
            raise NotImplementedError('text case not implemented')
        elif addition_embed_type == "text_image":
            raise NotImplementedError('text_image case not implemented')
        elif addition_embed_type == "text_time":
            self.add_time_proj = Timesteps('TimeSteps', addition_time_embed_dim, flip_sin_to_cos, freq_shift)   # type: ignore[arg-type]
            self.add_embedding = TimestepEmbedding('TimeStepEmbedding', projection_class_embeddings_input_dim, time_embed_dim) # type: ignore[arg-type]
        elif addition_embed_type == "image":
            raise NotImplementedError('image case not implemented')
        elif addition_embed_type == "image_hint":
            raise NotImplementedError('image_hint case not implemented')
        elif addition_embed_type is not None:
            raise ValueError(
                f"`addition_embed_type`: {addition_embed_type} must be None, 'text', 'text_image', 'text_time', 'image', or 'image_hint'."
            )

    def _set_pos_net_if_use_gligen(self, attention_type: str, cross_attention_dim: int):
        if attention_type in ["gated", "gated-text-image"]:
            raise NotImplementedError('gligen case not implemented')

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        # set recursively
        processors = {} # type: ignore[var-annotated]
        return processors

    def get_time_embed(
        self, sample: SimNN.SimTensor, timestep: Union[SimNN.SimTensor, float, int]
    ) -> Optional[SimNN.SimTensor]:
        timesteps = timestep
        assert sample.shape is not None, \
            "UNet2DConditionModel: input sample tensor shape must be set"
        # One timestep per batch element, matching upstream. This passed the bare
        # dimension before, which raised `Invalid shape type: <class 'int'>`.
        timesteps = ttsimF._from_shape(f'{self.name}_timesteps', shape=[sample.shape[0]])
        timesteps.set_module(self)
        t_emb = self.time_proj(timesteps)
        return t_emb

    def get_class_embed(self, sample: ttsimF.SimTensor, class_labels: Optional[ttsimF.SimTensor]):
        class_emb = None
        return class_emb
    
    def get_aug_embed(
        self, emb: SimNN.SimTensor, encoder_hidden_states: SimNN.SimTensor, added_cond_kwargs: Dict[str, Any]
    ) -> Optional[SimNN.SimTensor]:
        aug_emb = None
        return aug_emb

    def process_encoder_hidden_states(
        self, encoder_hidden_states: ttsimF.SimTensor, added_cond_kwargs: Dict[str, Any]
    ):
        return encoder_hidden_states

    def __call__(
        self,
        sample: SimNN.SimTensor,
        timestep: Union[SimNN.SimTensor, float, int],
        encoder_hidden_states: SimNN.SimTensor,
        class_labels: Optional[SimNN.SimTensor] = None,
        timestep_cond: Optional[SimNN.SimTensor] = None,
        attention_mask: Optional[SimNN.SimTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, SimNN.SimTensor]] = None,
        down_block_additional_residuals: Optional[Tuple[SimNN.SimTensor]] = None,
        mid_block_additional_residual: Optional[SimNN.SimTensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[SimNN.SimTensor]] = None,
        encoder_attention_mask: Optional[SimNN.SimTensor] = None,
        return_dict: bool = True,
    ):
        default_overall_up_factor = 2**self.num_upsamplers

        forward_upsample_size = False
        upsample_size = None

        assert sample.shape is not None, \
            "UNet2DConditionModel: input sample tensor shape must be set"
        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                forward_upsample_size = True
                break

        if attention_mask is not None:
            raise NotImplementedError('attention mask not implemented')

        if encoder_attention_mask is not None:
            raise NotImplementedError('encoder attention mask not implemented')

        # 0. center input if necessary
        if self.center_input_sample:
            sample = 2 * sample - 1.0   # type: ignore

        # 1. time
        t_emb = self.get_time_embed(sample=sample, timestep=timestep)
        emb = self.time_embedding(t_emb, timestep_cond)

        class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
        if class_emb is not None:
            raise NotImplementedError('class embedding not implemented')

        aug_emb = self.get_aug_embed(
            emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs   # type: ignore[arg-type]
        )
        if self.addition_embed_type == "image_hint":
            raise NotImplementedError('image hint embed type case not implemented')

        emb = emb + aug_emb if aug_emb is not None else emb

        encoder_hidden_states = self.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs    # type: ignore[arg-type]
        )

        # 2. pre-process
        sample = self.conv_in(sample)

        # 2.5 GLIGEN position net
        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            raise NotImplementedError('gligen case not implemented')

        # 3. down
        # we're popping the `scale` instead of getting it because otherwise `scale` will be propagated
        # to the internal blocks and will raise deprecation warnings. this will be confusing for our users.
        if cross_attention_kwargs is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            lora_scale = cross_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        # using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
        is_adapter = down_intrablock_additional_residuals is not None

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:   # type: ignore[union-attr]
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}
                if is_adapter and len(down_intrablock_additional_residuals) > 0:    # type: ignore[arg-type]
                    additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)  # type: ignore[union-attr]

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                if is_adapter and len(down_intrablock_additional_residuals) > 0:    # type: ignore[arg-type]
                    sample += down_intrablock_additional_residuals.pop(0)   # type: ignore[union-attr]

            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals # type: ignore[arg-type]
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual  # type: ignore[operator]
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)  # type: ignore[assignment]

            down_block_res_samples = new_down_block_res_samples # type: ignore[assignment]

        # 4. mid
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = self.mid_block(sample, emb)

            # To support T2I-Adapter-XL
            if (
                is_adapter
                and len(down_intrablock_additional_residuals) > 0   # type: ignore[arg-type]
                and sample.shape == down_intrablock_additional_residuals[0].shape   # type: ignore[index]
            ):
                sample += down_intrablock_additional_residuals.pop(0)   # type: ignore[union-attr]

        if is_controlnet:
            sample = sample + mid_block_additional_residual # type: ignore[operator]

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks): # type: ignore[arg-type]
            is_final_block = i == len(self.up_blocks) - 1   # type: ignore[arg-type]

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]    # type: ignore[attr-defined]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)] # type: ignore[assignment, attr-defined]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                res_sample = down_block_res_samples[-1]
                assert res_sample.shape is not None, \
                    "UNet2DConditionModel: down block residual sample tensor shape must be set"
                upsample_size = res_sample.shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )
        # 6. post-process
        self._tensors[sample.name] = sample
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return sample
