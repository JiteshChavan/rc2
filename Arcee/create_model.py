from Arcee_Models import Models
def create_model(config):
    assert "Arcee" in config.model or "Zigma" in config.model or "Vim" in config.model
    return Models[config.model](
        ssm_dstate = config.ssm_dstate,
        img_resolution = config.image_size // 8, # VAE latent space
        in_channels = config.num_in_channels,
        label_dropout = config.label_dropout, # cfg drop prob
        num_classes = config.num_classes,
        learn_sigma = config.learn_sigma,
        rms_norm = config.rms_norm,
        fused_add_norm=config.fused_add_norm,
        scan_type = config.scan_type,
        num_moe_experts = config.num_moe_experts,
        gated_linear_unit = config.gated_linear_unit,
        routing_mode = config.routing_mode,
        is_moe = config.is_moe,
        pe_type = config.pe_type,
        block_type = config.block_type, # Combined RC block
        learnable_pe = config.learnable_pe,
        drop_path = config.drop_path, # dropout feed forward
        use_final_norm = config.use_final_norm,
        use_attn_every_k_layers = config.use_attn_every_k_layers,
    )
    
    # residual_in_fp32 True
    # SSM cfg None
    # initializer_cfg None
    # mamba_moe_layers None
    # Add bias_linear False
    # use_independent_attn False
