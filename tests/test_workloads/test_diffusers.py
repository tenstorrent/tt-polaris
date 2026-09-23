#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for workloads/diffusers modules.
Tests focus on error handling, validation, and implemented features.
"""

import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest
else:
    try:
        import pytest
    except ImportError:
        pytest = None  # type: ignore[assignment]

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import ttsim.front.functional.op as F
from workloads.diffusers.attention import BasicTransformerBlock
from workloads.diffusers.attention_processor import Attention, AttnProcessor2_0
# Import diffusers modules
from workloads.diffusers.downsampling import Downsample2D
from workloads.diffusers.resnet import ResnetBlock2D
from workloads.diffusers.upsampling import Upsample2D


class TestDownsample2D:
    """Test Downsample2D module."""
    
    def test_initialization_with_conv(self):
        """Test successful initialization with use_conv=True."""
        downsample = Downsample2D(
            objname="test_downsample",
            channels=64,
            use_conv=True,
            out_channels=128
        )
        assert downsample.channels == 64
        assert downsample.out_channels == 128
        assert downsample.use_conv is True
    
    def test_non_conv_downsampling_not_implemented(self):
        """Test that non-convolutional downsampling raises NotImplementedError."""
        with pytest.raises(NotImplementedError) as exc_info:
            Downsample2D(
                objname="test_downsample",
                channels=64,
                use_conv=False,
                out_channels=64
            )
        
        assert "Non-convolutional downsampling" in str(exc_info.value)
        assert "use_conv=False" in str(exc_info.value)
    
    def test_channel_mismatch_for_non_conv(self):
        """Test that non-conv with mismatched channels raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Downsample2D(
                objname="test_downsample",
                channels=64,
                use_conv=False,
                out_channels=128  # Mismatch
            )
        
        assert "channels == out_channels" in str(exc_info.value)
        assert "64 != 128" in str(exc_info.value)
    
    def test_rms_norm_not_supported(self):
        """Test that RMSNorm raises NotImplementedError."""
        with pytest.raises(NotImplementedError) as exc_info:
            Downsample2D(
                objname="test_downsample",
                channels=64,
                use_conv=True,
                norm_type="rms_norm"
            )
        
        assert "RMSNorm not supported" in str(exc_info.value)
    
    def test_forward_with_norm_not_implemented(self):
        """Test that forward pass with normalization raises NotImplementedError."""
        downsample = Downsample2D(
            objname="test_downsample",
            channels=64,
            use_conv=True,
            norm_type="ln_norm"
        )
        
        # Create mock input tensor
        hidden_states = F._from_shape('input', [1, 64, 32, 32])
        
        with pytest.raises(NotImplementedError) as exc_info:
            downsample(hidden_states)
        
        assert "Normalization in Downsample2D forward pass" in str(exc_info.value)
    
    def test_forward_channel_validation(self):
        """Test that forward pass validates input channels."""
        downsample = Downsample2D(
            objname="test_downsample",
            channels=64,
            use_conv=True
        )
        
        # Create input with wrong number of channels
        wrong_input = F._from_shape('input', [1, 32, 32, 32])  # 32 instead of 64
        
        with pytest.raises(ValueError) as exc_info:
            downsample(wrong_input)
        
        assert "channel mismatch" in str(exc_info.value).lower()
        assert "Expected 64" in str(exc_info.value)
        assert "got 32" in str(exc_info.value)


class TestUpsample2D:
    """Test Upsample2D module."""
    
    def test_initialization_basic(self):
        """Test basic initialization."""
        upsample = Upsample2D(
            objname="test_upsample",
            channels=64,
            use_conv=True
        )
        assert upsample.channels == 64
        assert upsample.use_conv is True
    
    def test_layernorm_not_supported(self):
        """Test that LayerNorm raises NotImplementedError."""
        with pytest.raises(NotImplementedError) as exc_info:
            Upsample2D(
                objname="test_upsample",
                channels=64,
                norm_type="ln_norm"
            )
        
        assert "LayerNorm not supported" in str(exc_info.value)
    
    def test_rms_norm_not_supported(self):
        """Test that RMSNorm raises NotImplementedError."""
        with pytest.raises(NotImplementedError) as exc_info:
            Upsample2D(
                objname="test_upsample",
                channels=64,
                norm_type="rms_norm"
            )
        
        assert "RMSNorm not supported" in str(exc_info.value)
    
    def test_conv_transpose_not_supported(self):
        """Test that ConvTranspose2d raises NotImplementedError."""
        with pytest.raises(NotImplementedError) as exc_info:
            Upsample2D(
                objname="test_upsample",
                channels=64,
                use_conv_transpose=True
            )
        
        assert "ConvTranspose2d not supported" in str(exc_info.value)
    
    def test_forward_channel_validation(self):
        """Test that forward pass validates input channels."""
        upsample = Upsample2D(
            objname="test_upsample",
            channels=64,
            use_conv=True
        )
        
        # Create input with wrong number of channels
        wrong_input = F._from_shape('input', [1, 32, 16, 16])
        
        with pytest.raises(ValueError) as exc_info:
            upsample(wrong_input)
        
        assert "channel mismatch" in str(exc_info.value).lower()
        assert "Expected 64" in str(exc_info.value)
        assert "got 32" in str(exc_info.value)


class TestResnetBlock2D:
    """Test ResnetBlock2D module."""
    
    def test_initialization_basic(self):
        """Test basic initialization."""
        resnet = ResnetBlock2D(
            objname="test_resnet",
            in_channels=64,
            out_channels=64
        )
        assert resnet.in_channels == 64
        assert resnet.out_channels == 64
    
    def test_fir_upsampling_not_supported(self):
        """Test that FIR upsampling raises NotImplementedError."""
        with pytest.raises(NotImplementedError) as exc_info:
            ResnetBlock2D(
                objname="test_resnet",
                in_channels=64,
                up=True,
                kernel="fir"  # type: ignore[arg-type]
            )
        
        assert "FIR upsampling not supported" in str(exc_info.value)
    
    def test_sde_vp_upsampling_not_supported(self):
        """Test that SDE_VP upsampling raises NotImplementedError."""
        with pytest.raises(NotImplementedError) as exc_info:
            ResnetBlock2D(
                objname="test_resnet",
                in_channels=64,
                up=True,
                kernel="sde_vp"  # type: ignore[arg-type]
            )
        
        assert "SDE_VP upsampling not supported" in str(exc_info.value)
    
    def test_fir_downsampling_not_supported(self):
        """Test that FIR downsampling raises NotImplementedError."""
        with pytest.raises(NotImplementedError) as exc_info:
            ResnetBlock2D(
                objname="test_resnet",
                in_channels=64,
                down=True,
                kernel="fir"  # type: ignore[arg-type]
            )
        
        assert "FIR downsampling not supported" in str(exc_info.value)
    
    def test_sde_vp_downsampling_not_supported(self):
        """Test that SDE_VP downsampling raises NotImplementedError."""
        with pytest.raises(NotImplementedError) as exc_info:
            ResnetBlock2D(
                objname="test_resnet",
                in_channels=64,
                down=True,
                kernel="sde_vp"  # type: ignore[arg-type]
            )
        
        assert "SDE_VP downsampling not supported" in str(exc_info.value)
    
    def test_scale_shift_not_implemented(self):
        """Test that scale_shift time embedding norm raises NotImplementedError during forward."""
        resnet = ResnetBlock2D(
            objname="test_resnet",
            in_channels=64,
            out_channels=64,
            time_embedding_norm="scale_shift"
        )
        
        input_tensor = F._from_shape('input', [1, 64, 32, 32])
        temb = F._from_shape('temb', [1, 512])
        
        with pytest.raises(NotImplementedError) as exc_info:
            resnet(input_tensor, temb)
        
        assert "scale_shift not implemented" in str(exc_info.value)


class TestBasicTransformerBlock:
    """Test BasicTransformerBlock module."""
    
    def test_gated_attention_not_supported(self):
        """Test that gated attention raises NotImplementedError."""
        with pytest.raises(NotImplementedError) as exc_info:
            BasicTransformerBlock(
                objname="test_transformer",
                dim=512,
                num_attention_heads=8,
                attention_head_dim=64,
                attention_type="gated"
            )
        
        assert "Gated Self Attention not supported" in str(exc_info.value)
    
    def test_ada_norm_single_not_supported(self):
        """Test that ada_norm_single raises NotImplementedError."""
        with pytest.raises(NotImplementedError) as exc_info:
            BasicTransformerBlock(
                objname="test_transformer",
                dim=512,
                num_attention_heads=8,
                attention_head_dim=64,
                norm_type="ada_norm_single"
            )
        
        assert "Single AdaNorm not supported" in str(exc_info.value)
    
    def test_chunked_feed_forward_not_supported(self):
        """Test that chunked feed forward raises NotImplementedError during forward."""
        block = BasicTransformerBlock(
            objname="test_transformer",
            dim=512,
            num_attention_heads=8,
            attention_head_dim=64,
            norm_type="layer_norm"
        )
        
        # Set chunk size to trigger the error
        block._chunk_size = 256  # type: ignore[assignment]
        
        hidden_states = F._from_shape('hidden', [2, 77, 512])
        
        with pytest.raises(NotImplementedError) as exc_info:
            block(hidden_states)
        
        assert "Chunked feed forward not supported" in str(exc_info.value)


class TestAttention:
    """Test Attention module."""
    
    def test_initialization_basic(self):
        """Test basic initialization."""
        attention = Attention(
            objname="test_attention",
            query_dim=512,
            cross_attention_dim=512,
            heads=8,
            dim_head=64
        )
        assert attention.heads == 8
        assert attention.inner_dim == 512  # heads * dim_head
    
    def test_spatial_norm_not_supported(self):
        """Test that spatial norm raises NotImplementedError."""
        with pytest.raises(NotImplementedError) as exc_info:
            Attention(
                objname="test_attention",
                query_dim=512,
                heads=8,
                dim_head=64,
                spatial_norm_dim=512
            )
        
        assert "spatial norm not supported" in str(exc_info.value)
    
    def test_qk_norm_not_supported(self):
        """Test that qk_norm raises NotImplementedError."""
        with pytest.raises(NotImplementedError) as exc_info:
            Attention(
                objname="test_attention",
                query_dim=512,
                heads=8,
                dim_head=64,
                qk_norm="layer_norm"
            )
        
        assert "qk_norm not None is not supported yet" in str(exc_info.value)


class TestAttnProcessor2_0:
    """Test AttnProcessor2_0 module."""
    
    def test_4d_input_not_supported(self):
        """Test that 4D input raises NotImplementedError."""
        # Create a mock attention module
        attn = Attention(
            objname="test_attention",
            query_dim=512,
            cross_attention_dim=512,
            heads=8,
            dim_head=64
        )
        
        processor = AttnProcessor2_0(attn)
        
        # Create 4D input
        hidden_states_4d = F._from_shape('hidden', [1, 512, 16, 16])
        
        with pytest.raises(NotImplementedError) as exc_info:
            processor(attn, hidden_states_4d)
        
        assert "only works for 3 dims" in str(exc_info.value)
    
    def test_attention_mask_not_supported(self):
        """Test that attention mask raises NotImplementedError."""
        attn = Attention(
            objname="test_attention",
            query_dim=512,
            cross_attention_dim=512,
            heads=8,
            dim_head=64
        )
        
        processor = AttnProcessor2_0(attn)
        
        # Create 3D input (valid)
        hidden_states = F._from_shape('hidden', [2, 77, 512])
        attention_mask = F._from_shape('mask', [2, 77])
        
        with pytest.raises(NotImplementedError) as exc_info:
            processor(attn, hidden_states, attention_mask=attention_mask)
        
        assert "attention mask not supported" in str(exc_info.value)
    
    def test_group_norm_not_supported(self):
        """Test that group norm raises NotImplementedError."""
        attn = Attention(
            objname="test_attention",
            query_dim=512,
            cross_attention_dim=512,
            heads=8,
            dim_head=64,
            norm_num_groups=32
        )
        
        processor = AttnProcessor2_0(attn)
        
        hidden_states = F._from_shape('hidden', [2, 77, 512])
        
        with pytest.raises(NotImplementedError) as exc_info:
            processor(attn, hidden_states)
        
        assert "group norm not supported" in str(exc_info.value)


class TestIntegration:
    """Integration tests for diffusers modules."""
    
    def test_stable_diffusion_model_exists(self):
        """Test that the stable diffusion integration test file exists."""
        sd_test_path = os.path.join(
            os.path.dirname(__file__),
            '../../workloads/diffusers/stablediffusion.py'
        )
        assert os.path.exists(sd_test_path), "Stable diffusion test file should exist"
    
    def test_module_imports(self):
        """Test that all diffusers modules can be imported."""
        try:
            from workloads.diffusers import (activations, attention, attention_processor, autoencoder_kl,  # noqa: F401
                                             downsampling, embeddings, resnet, transformer_2d, unet_2d_blocks,
                                             unet_2d_condition, upsampling, vae)
        except ImportError as e:
            pytest.fail(f"Failed to import diffusers modules: {e}")

    @pytest.mark.unit
    def test_stable_diffusion_forward(self):
        """The SD forward runs end to end and produces a 512x512 image.

        UNet2DConditionModel.get_time_embed built its timestep tensor from the
        bare batch dimension, which raised `Invalid shape type: <class 'int'>`,
        so the workload never ran. The timestep tensor must have shape [B].
        """
        from workloads.diffusers.stablediffusion import SDModel

        sd_model = SDModel('sd', {'bs': 1})
        sd_model.create_input_tensors()
        latents = sd_model.input_tensors['latents']
        assert latents.shape is not None
        timesteps = F._from_shape('t', shape=[])
        t_emb = sd_model.unet.get_time_embed(latents, timesteps)
        assert t_emb is not None
        assert t_emb.shape is not None and t_emb.shape[0] == latents.shape[0], 'one timestep per batch element'

        image = sd_model()
        assert image.shape == [1, 3, 512, 512]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

