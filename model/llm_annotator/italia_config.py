from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig

class ItaliaConfig(GPTNeoXConfig):
    model_type = "italia"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        hidden_act="gelu_new",
        *args,
        **kwargs,
    ):
        super().__init__(
            hidden_act=hidden_act,
            *args,
            **kwargs,
        )