from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

from .model import ModelConfig, RealDataMoEForCausalLM


try:
    from lm_eval import utils
    from lm_eval.api.model import TemplateLM
    from lm_eval.api.registry import register_model
except ImportError as exc:  # pragma: no cover - exercised on remote eval env.
    raise ImportError(
        "lm-evaluation-harness is required for RealDataMoELM. "
        "Install/import lm_eval before using this wrapper."
    ) from exc


def _pad_right(sequences: List[torch.Tensor], pad_token_id: int) -> torch.Tensor:
    max_len = max(sequence.numel() for sequence in sequences)
    output = sequences[0].new_full((len(sequences), max_len), pad_token_id)
    for index, sequence in enumerate(sequences):
        output[index, : sequence.numel()] = sequence
    return output


@register_model("real_data_moe")
class RealDataMoELM(TemplateLM):
    """lm-evaluation-harness adapter for the custom Common/Tail MoE checkpoint."""

    AUTO_MODEL_CLASS = None

    def __init__(
        self,
        checkpoint: str,
        tokenizer: str,
        device: str = "cuda:0",
        batch_size: Union[int, str] = 1,
        max_batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        dtype: str = "bfloat16",
        **_: object,
    ) -> None:
        super().__init__()
        self._device = torch.device(device)
        self.batch_size_per_gpu = int(batch_size) if batch_size != "auto" else 1
        self.max_batch_size = max_batch_size
        self.backend = "causal"
        self.logits_cache = True
        self.softmax_dtype = torch.float32
        self.add_bos_token = False
        self._dtype_name = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._tokenizer_name = tokenizer

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        config = ModelConfig(**payload["model_config"])
        self._max_length = int(max_length or config.max_position_embeddings)
        self.model = RealDataMoEForCausalLM(config)
        self.model.load_state_dict(payload["model"])
        self.model.eval().to(self._device)
        del payload

        if dtype in {"bfloat16", "bf16"}:
            self._autocast_dtype = torch.bfloat16
        elif dtype in {"float16", "fp16"}:
            self._autocast_dtype = torch.float16
        elif dtype in {"float32", "fp32", "none"}:
            self._autocast_dtype = None
        else:
            raise ValueError("dtype must be bfloat16, float16, float32, or none")

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def batch_size(self) -> int:
        return self.batch_size_per_gpu

    @property
    def eot_token_id(self) -> int:
        token_id = self.tokenizer.eos_token_id
        if token_id is None:
            token_id = self.tokenizer.pad_token_id
        return int(token_id if token_id is not None else 0)

    @property
    def tokenizer_name(self) -> str:
        return self._tokenizer_name

    def tok_encode(
        self,
        string: str,
        left_truncate_len: Optional[int] = None,
        add_special_tokens: Optional[bool] = None,
        **_: object,
    ) -> List[int]:
        if add_special_tokens is None:
            add_special_tokens = False
        tokens = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len is not None:
            tokens = tokens[-left_truncate_len:]
        return tokens

    def tok_decode(self, tokens, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _model_call(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self._autocast_dtype is not None and self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=self._autocast_dtype):
                    logits, _ = self.model(input_ids)
            else:
                logits, _ = self.model(input_ids)
        return logits.float()

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Optional[Tuple[str, str]], List[int], List[int]]],
        disable_tqdm: bool = False,
        override_bs: Optional[int] = None,
    ) -> List[Tuple[float, bool]]:
        results: List[Tuple[float, bool]] = []
        batch_size = int(override_bs or self.batch_size)
        iterator = range(0, len(requests), batch_size)

        for start in tqdm(
            iterator,
            total=(len(requests) + batch_size - 1) // batch_size,
            disable=disable_tqdm,
            desc="Running loglikelihood requests",
        ):
            chunk = requests[start : start + batch_size]
            inputs = []
            input_lengths = []
            continuation_tokens = []

            for _, context_enc, continuation_enc in chunk:
                if len(context_enc) == 0:
                    context_enc = [self.prefix_token_id]
                if len(continuation_enc) == 0:
                    continuation_enc = [self.eot_token_id]
                if len(continuation_enc) > self.max_length:
                    continuation_enc = continuation_enc[-self.max_length :]

                input_tokens = (context_enc + continuation_enc)[-(self.max_length + 1) :][
                    :-1
                ]
                tensor = torch.tensor(input_tokens, dtype=torch.long, device=self.device)
                inputs.append(tensor)
                input_lengths.append(tensor.numel())
                continuation_tokens.append(continuation_enc)

            batched_inputs = _pad_right(inputs, self.eot_token_id)
            log_probs = F.log_softmax(
                self._model_call(batched_inputs), dim=-1, dtype=self.softmax_dtype
            )

            for (request_str, _, _), row, inplen, cont_toks in zip(
                chunk, log_probs, input_lengths, continuation_tokens
            ):
                contlen = len(cont_toks)
                logits = row[inplen - contlen : inplen]
                cont = torch.tensor(cont_toks, dtype=torch.long, device=self.device)
                greedy = bool((logits.argmax(dim=-1) == cont).all().item())
                score = float(logits.gather(1, cont.unsqueeze(-1)).sum().item())
                answer = (score, greedy)
                results.append(answer)
                if request_str is not None:
                    self.cache_hook.add_partial("loglikelihood", request_str, answer)

        return results

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> List[float]:
        all_outputs: List[float] = []
        for (string,) in tqdm(
            [req.args for req in requests],
            disable=disable_tqdm,
            desc="Running rolling loglikelihood requests",
        ):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )
            rolling_requests = [(None,) + window for window in rolling_token_windows]
            outputs = self._loglikelihood_tokens(
                rolling_requests, disable_tqdm=True, override_bs=self.batch_size
            )
            total = sum(output[0] for output in outputs)
            all_outputs.append(total)
            self.cache_hook.add_partial("loglikelihood_rolling", (string,), total)
        return all_outputs

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        outputs: List[str] = []
        for context, gen_kwargs in tqdm(
            [req.args for req in requests],
            disable=disable_tqdm,
            desc="Running generate_until requests",
        ):
            until = gen_kwargs.get("until", [])
            if isinstance(until, str):
                until = [until]
            max_gen_toks = int(gen_kwargs.get("max_gen_toks", 256))

            tokens = self.tok_encode(context)[-self.max_length :]
            generated: List[int] = []
            for _ in range(max_gen_toks):
                input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
                logits = self._model_call(input_ids)
                next_token = int(logits[0, -1].argmax(dim=-1).item())
                tokens.append(next_token)
                tokens = tokens[-self.max_length :]
                generated.append(next_token)
                text = self.tok_decode(generated)
                if any(stop and stop in text for stop in until):
                    for stop in until:
                        if stop and stop in text:
                            text = text.split(stop)[0]
                            break
                    break
            else:
                text = self.tok_decode(generated)

            outputs.append(text)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text)
        return outputs
