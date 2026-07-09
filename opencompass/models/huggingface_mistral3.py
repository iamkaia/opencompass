from typing import Optional

from mmengine.device import is_npu_available

from opencompass.models.huggingface_above_v4_33 import (
    HuggingFacewithChatTemplate,
    _set_model_kwargs_torch_dtype,
)
from opencompass.registry import MODELS


@MODELS.register_module()
class HuggingFaceMistral3(HuggingFacewithChatTemplate):
    """Chat wrapper for multimodal Mistral 3 checkpoints in text-only eval."""

    def _load_tokenizer(self,
                        path: str,
                        kwargs: dict,
                        pad_token_id: Optional[int] = None):
        try:
            return super()._load_tokenizer(path, kwargs, pad_token_id)
        except ValueError as exc:
            # Ministral-3-*-2512 publishes a Transformers 5 tokenizer config
            # (`TokenizersBackend`). Transformers 4.57 already has a compatible
            # Rust-tokenizers implementation, but AutoTokenizer cannot resolve
            # that new class name. Loading the fast tokenizer directly bypasses
            # only the class-name lookup and keeps the published tokenizer.json.
            if 'Tokenizer class TokenizersBackend does not exist' not in str(exc):
                raise

        from pathlib import Path

        from transformers import PreTrainedTokenizerFast
        from transformers.utils.hub import cached_file

        tokenizer_file = cached_file(
            path, 'tokenizer.json', trust_remote_code=True)
        chat_template_file = cached_file(
            path, 'chat_template.jinja', trust_remote_code=True)
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_file,
            bos_token='<s>',
            eos_token='</s>',
            pad_token='<pad>',
            unk_token='<unk>',
            chat_template=Path(chat_template_file).read_text(),
        )
        self.tokenizer.padding_side = kwargs.get('padding_side', 'left')
        self.tokenizer.truncation_side = kwargs.get('truncation_side', 'left')

        if pad_token_id is not None:
            self.tokenizer.pad_token_id = pad_token_id
        elif self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _load_model(self,
                    path: str,
                    kwargs: dict,
                    peft_path: Optional[str] = None,
                    peft_kwargs: dict = dict()):
        from transformers import AutoModelForImageTextToText

        model_kwargs = dict(device_map='auto', trust_remote_code=True)
        model_kwargs.update(kwargs)
        model_kwargs = _set_model_kwargs_torch_dtype(model_kwargs)
        if is_npu_available():
            model_kwargs['device_map'] = 'npu'

        self.model = AutoModelForImageTextToText.from_pretrained(
            path, **model_kwargs)

        if peft_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model, peft_path, is_trainable=False, **peft_kwargs)

        self.model.eval()
        self.model.generation_config.do_sample = False
