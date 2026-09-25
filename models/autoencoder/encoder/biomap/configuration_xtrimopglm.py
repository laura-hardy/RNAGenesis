from transformers import PretrainedConfig


class xTrimoPGLMConfig(PretrainedConfig):
    model_type = "xTrimoPGLM"
    def __init__(
        self,
        num_layers=28,
        padded_vocab_size=128,
        hidden_size=4096,
        ffn_hidden_size=6832,
        kv_channels=64,
        num_attention_heads=40,
        seq_length=2048,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        layernorm_epsilon=1e-5,
        initializer_range=0.02,
        glu_activation='geglu',
        rmsnorm=False,
        deepnorm=True,
        apply_residual_connection_post_layernorm=True,
        post_layer_norm=True,
        add_bias_linear=True,
        add_qkv_bias=True,
        bias_dropout_fusion=True,
        multi_query_attention=False,
        multi_query_group_num=1,
        apply_query_key_layer_scaling=True,
        attention_softmax_in_fp32=True,
        fp32_residual_connection=False,
        quantization_bit=0,
        rotary_embedding_2d=False,
        use_pytorch_sdpa=True,
        is_causal=False,
        use_cache=True,
        moe=False,
        num_experts=0, 
        experts_per_token=0,
        untie_head=False,
        head_num=1,
        **kwargs
    ):

        if not deepnorm and apply_residual_connection_post_layernorm:
            print(f"Warning: deepnorm is False and apply_residual_connection_post_layernorm is True")

        if deepnorm:
            apply_residual_connection_post_layernorm = True

        self.num_layers = num_layers
        self.vocab_size = padded_vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.kv_channels = kv_channels
        self.num_attention_heads = num_attention_heads
        self.seq_length = seq_length
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.layernorm_epsilon = layernorm_epsilon
        self.glu_activation = glu_activation
        self.initializer_range = initializer_range
        self.rmsnorm = rmsnorm
        self.deepnorm = deepnorm
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.post_layer_norm = post_layer_norm
        self.add_bias_linear = add_bias_linear
        self.add_qkv_bias = add_qkv_bias
        self.bias_dropout_fusion = bias_dropout_fusion
        self.multi_query_attention = multi_query_attention
        self.multi_query_group_num = multi_query_group_num
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.fp32_residual_connection = fp32_residual_connection
        self.quantization_bit = quantization_bit
        self.rotary_embedding_2d = rotary_embedding_2d
        self.is_causal = is_causal
        self.use_cache=use_cache
        self.use_pytorch_sdpa = use_pytorch_sdpa
        self.moe = moe
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.untie_head = untie_head
        self.head_num=head_num
        super().__init__(**kwargs)