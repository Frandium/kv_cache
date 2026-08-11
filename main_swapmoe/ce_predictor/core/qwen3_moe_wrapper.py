# qwen3_moe_wrapper.py
import torch
import torch.nn as nn
import os
from transformers import AutoModelForCausalLM

class Qwen15MoEForPredictor(nn.Module):
    """
    这是一个很薄的“外壳”：
    - 内部是 HF 的 Qwen1.5-MoE-A2.7B
    - 外壳用 hook 抓每层的 layer_input / attn_output / expert_label / value_states
    """
    def __init__(self, model_path, device="cuda:0", dtype=torch.bfloat16):
        super().__init__()
        model_path = os.path.abspath(model_path)
        assert os.path.isdir(model_path), f"Model dir not found: {model_path}"
        assert os.path.exists(os.path.join(model_path, "config.json")), f"config.json not in: {model_path}"
        self.base = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map={"": device},
            trust_remote_code=True,
            local_files_only=True,
        )
        self.device = device

        self.layer_inputs = []
        self.attn_outputs = []
        self.expert_labels = []
        self.value_states = []  # ★★★ 新增：用于存储 V 矩阵输出 ★★★

        self._register_hooks()

    def _register_hooks(self):
        layers = self.base.model.layers  # Qwen1.5 的 transformer blocks

        cfg = self.base.config
        self.moe_top_k = getattr(cfg, "moe_top_k", getattr(cfg, "num_experts_per_tok", 4))
        self.hidden_size = cfg.hidden_size

        # 1) 抓 layer input
        def pre_hook(module, inputs):
            x = inputs[0]
            self.layer_inputs.append(x.detach())

        # 2) 抓 self-attn 的输出（未加残差）
        def attn_hook(module, inputs, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            self.attn_outputs.append(out.detach())

        # 3) ★★★ 新增：抓 self-attn 内部 v_proj 的输出 ★★★
        def v_proj_hook(module, inputs, output):
            # output shape: [B, T, num_key_value_heads * head_dim]
            # 注意：如果是 GQA (Grouped Query Attention)，这里的维度可能小于 hidden_size
            self.value_states.append(output.detach())

        # 4) 直接在 MoE-MLP 上算 router top-k expert indices
        def moe_mlp_hook(module, inputs, output):
            x = inputs[0]
            gate = (
                getattr(module, "gate", None)
                or getattr(module, "router", None)
                or getattr(module, "gate_proj", None)
            )
            if gate is None:
                return

            B, T, H = x.shape
            scores = gate(x)
            if scores.dim() == 2:
                scores = scores.view(B, T, -1)

            idx = scores.topk(self.moe_top_k, dim=-1).indices
            self.expert_labels.append(idx.detach())

        for layer in layers:
            layer.register_forward_pre_hook(pre_hook)

            attn_mod = getattr(layer, "self_attn", None) or getattr(layer, "attn", None) or getattr(layer, "attention", None)
            if attn_mod is not None:
                attn_mod.register_forward_hook(attn_hook)
                
                # ★★★ 注册 v_proj hook ★★★
                # Qwen2/Qwen1.5 的结构通常包含 q_proj, k_proj, v_proj, o_proj
                v_proj = getattr(attn_mod, "v_proj", None)
                if v_proj is not None:
                    v_proj.register_forward_hook(v_proj_hook)

            mlp_mod = getattr(layer, "mlp", None)
            if mlp_mod is not None:
                mlp_mod.register_forward_hook(moe_mlp_hook)

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        output_layer_input=False,
        output_attn_output=False,
        output_expert_label=False,
        output_value_states=False,  # ★★★ 新增参数 ★★★
        **kwargs
    ):
        # 每次 forward 先清空
        self.layer_inputs = []
        self.attn_outputs = []
        self.expert_labels = []
        self.value_states = []  # ★★★ 清空 V 缓存 ★★★

        _ = self.base(input_ids=input_ids, **kwargs)

        out = {}
        if output_layer_input:
            out["layer_input"] = self.layer_inputs
        if output_attn_output:
            out["attn_output"] = self.attn_outputs
        if output_expert_label:
            out["expert_label"] = self.expert_labels
        
        # ★★★ 返回 V 矩阵 ★★★
        if output_value_states:
            out["value_states"] = self.value_states
            
        return out

class Qwen3MoEForPredictor(nn.Module):
    """
    输出对齐你原来 Qwen15 wrapper 的格式：
      - out["layer_input"]  : List[(B,S,H)] 每层输入
      - out["attn_output"]  : List[(B,S,H)] 每层 self-attn 输出(未加残差)
      - out["expert_label"] : List[(B,S,K)] 每层 router topk experts
    """

    def __init__(self, model_path, device="cuda:0", dtype=torch.bfloat16, device_map="auto"):
        super().__init__()
        model_path = os.path.abspath(model_path)
        assert os.path.isdir(model_path), f"Model dir not found: {model_path}"
        assert os.path.exists(os.path.join(model_path, "config.json")), f"config.json not in: {model_path}"

        self.base = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=False,   # Qwen3 是 transformers 原生
            local_files_only=True,
        )
        self.base.eval()

        self.device = device
        self.gather_device = torch.device(device)

        cfg = self.base.config
        self.moe_top_k = int(getattr(cfg, "num_experts_per_tok", 8))
        self.hidden_size = int(getattr(cfg, "hidden_size", 0))
        self.num_heads = int(getattr(cfg, "num_attention_heads", 0))
        self.num_kv_heads = int(getattr(cfg, "num_key_value_heads", getattr(cfg, "num_kv_heads", self.num_heads)))
        assert self.num_heads > 0, "num_attention_heads missing in config"
        self.head_dim = self.hidden_size // self.num_heads
        assert self.head_dim * self.num_heads == self.hidden_size, "hidden_size not divisible by num_attention_heads"

        self.value_states = []   # ★ 新增

        self.layer_inputs = []
        self.attn_outputs = []
        self.expert_labels = []

        self._register_hooks()

    def _get_layers(self):
        # 适配常见结构：base.model.layers
        if hasattr(self.base, "model") and hasattr(self.base.model, "layers"):
            return self.base.model.layers
        raise RuntimeError("Cannot find decoder layers at base.model.layers")

    def _input_device(self):
        # device_map="auto" 时把 input_ids 放到第一个参数所在设备
        return next(self.base.parameters()).device

    def _register_hooks(self):
        layers = self._get_layers()

        def pre_hook(module, inputs):
            x = inputs[0]  # (B,S,H)
            self.layer_inputs.append(x.detach().to(self.gather_device))

        def attn_hook(module, inputs, output):
            # MultiheadAttention/自定义attention可能返回 tuple
            out = output[0] if isinstance(output, (tuple, list)) else output
            self.attn_outputs.append(out.detach().to(self.gather_device))

        def value_pre_hook(attn_mod, args, kwargs):
            # Qwen3 这类实现经常用 kwargs 传 hidden_states
            x = args[0] if (args is not None and len(args) > 0) else None
            if x is None:
                x = kwargs.get("hidden_states", None)
            if x is None:
                x = kwargs.get("x", None)
            if x is None:
                return

            v_proj = getattr(attn_mod, "v_proj", None) or getattr(attn_mod, "value_proj", None)
            if v_proj is None:
                return

            v = v_proj(x)  # (B,S,kv_heads*head_dim) or (B,S,H)
            if v.dim() != 3:
                return

            B, S, kv_dim = v.shape
            H = self.hidden_size

            # 1) head_dim：优先从模块拿；拿不到就用 config 推
            head_dim = getattr(attn_mod, "head_dim", None)
            if head_dim is None:
                cfg_heads = int(getattr(self.base.config, "num_attention_heads", 0))
                assert cfg_heads > 0, "config missing num_attention_heads"
                head_dim = H // cfg_heads
            head_dim = int(head_dim)

            # 2) num_heads：强制用 hidden_size/head_dim（保证与 H 对齐）
            assert H % head_dim == 0, f"hidden_size {H} not divisible by head_dim {head_dim}"
            num_heads = H // head_dim

            # 1) 非 GQA：kv_dim == H，直接用
            if kv_dim == H:
                self.value_states.append(v.detach().to(self.gather_device))
                return

            # 2) GQA：kv_dim = kv_heads * head_dim
            if kv_dim % head_dim != 0:
                raise RuntimeError(f"Unexpected V dim {kv_dim}; not divisible by head_dim {head_dim}")

            kv_heads = kv_dim // head_dim
            if num_heads % kv_heads != 0:
                raise RuntimeError(f"num_heads {num_heads} not divisible by kv_heads {kv_heads}")

            # (B,S,kv_heads,head_dim) -> repeat 到 (B,S,num_heads,head_dim) -> (B,S,H)
            v = v.view(B, S, kv_heads, head_dim)
            repeat = num_heads // kv_heads
            v = v.repeat_interleave(repeat, dim=2).reshape(B, S, H)

            self.value_states.append(v.detach().to(self.gather_device))

        for layer in layers:
            layer.register_forward_pre_hook(pre_hook)

            attn_mod = getattr(layer, "self_attn", None) or getattr(layer, "attn", None) or getattr(layer, "attention", None)
            if attn_mod is not None:
                attn_mod.register_forward_pre_hook(value_pre_hook, with_kwargs=True)  # ★ 新增：抓 V
                attn_mod.register_forward_hook(attn_hook)

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        output_layer_input=False,
        output_attn_output=False,
        output_expert_label=False,
        output_value_states=False,    # ★ 新增
        **kwargs
    ):
        # 清空缓存
        self.layer_inputs = []
        self.attn_outputs = []
        self.expert_labels = []
        self.value_states = []

        input_ids = input_ids.to(self._input_device())

        out = self.base(
            input_ids=input_ids,
            use_cache=False,
            output_router_logits=output_expert_label,  # 关键：拿 router logits
            return_dict=True,
            **kwargs,
        )

        if output_expert_label:
            B, S = input_ids.shape
            router_logits = out.router_logits  # tuple(len=L)，每个可能是 (B,S,E) 或 (B*S,E)
            labels = []
            for scores in router_logits:
                if scores.dim() == 2:
                    scores = scores.view(B, S, -1)
                idx = torch.topk(scores, k=self.moe_top_k, dim=-1).indices  # (B,S,K)
                labels.append(idx.detach().to(self.gather_device))
            self.expert_labels = labels

        ret = {}
        if output_layer_input:
            ret["layer_input"] = self.layer_inputs
        if output_attn_output:
            ret["attn_output"] = self.attn_outputs
        if output_expert_label:
            ret["expert_label"] = self.expert_labels
        if output_value_states:
            ret["value_states"] = self.value_states
        return ret


class Qwen3NextMoEForPredictor(nn.Module):
    """
    面向 Qwen3-Next / Qwen3 系列的通用 wrapper：
      - out["layer_input"]  : List[(B,S,H)] len=L
      - out["attn_output"]  : List[(B,S,H)] len=L   ✅ 强保证（尽最大努力）
      - out["expert_label"] : List[(B,S,K)] len=L   (来自 output_router_logits)
      - out["value_states"] : List[(B,S,H)] len=L   (可选，来自 v_proj；若找不到则 None)

    关键点：
      1) 对每个 layer 按 index 定点写入（不 append），避免“只抓到 12 个”的错位
      2) attention 模块用 named_modules() 递归搜索 + 启发式筛选
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",                 # gather_device：把 hook 结果搬到哪张卡
        dtype: torch.dtype = torch.bfloat16,
        device_map="auto",
        trust_remote_code: bool = True,         # Qwen3-Next 通常需要 True（你已经能 load 说明也可）
        local_files_only: bool = True,
        strict_attn: bool = True,               # True: 如果某层找不到 attn 模块就报错；False: 填 None
        verbose: bool = True,                   # 打印一次找到的 attn module 名称
    ):
        super().__init__()
        model_path = os.path.abspath(model_path)
        assert os.path.isdir(model_path), f"Model dir not found: {model_path}"
        assert os.path.exists(os.path.join(model_path, "config.json")), f"config.json not in: {model_path}"

        n_gpus = torch.cuda.device_count()
        assert n_gpus >= 8, f"need >=8 gpus, got {n_gpus}"

        max_memory = {i: "90GiB" for i in range(n_gpus)}

        # 你要的策略：predictor 在 cuda0，gather_device 在 cuda1
        # -> 给 0/1 留空间，不要让模型占满
        max_memory[0] = "40GiB"   # 给 predictor 留 >= 20GiB（按需调）
        max_memory[1] = "40GiB"   # 给 gather 留空间（按需调）

        self.base = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,                         # 新版 transformers 推荐 dtype
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        self.base.eval()
        print("[DEBUG] hf_device_map exists?", hasattr(self.base, "hf_device_map"))
        if hasattr(self.base, "hf_device_map"):
            # 看看 cuda:0 上到底放了哪些模块
            n0 = sum(1 for _, d in self.base.hf_device_map.items() if str(d) in ["0", "cuda:0"])
            n1 = sum(1 for _, d in self.base.hf_device_map.items() if str(d) in ["1", "cuda:1"])
            print(f"[DEBUG] mapped modules on cuda0={n0}, cuda1={n1}")
            # 打印前 20 个映射（确认不是全塞 0）
            for k, v in list(self.base.hf_device_map.items())[:20]:
                print("[DEBUG] map", k, "->", v)

        print("[DEBUG] mem cuda0 GiB:", torch.cuda.memory_allocated(0)/1024**3)

        self.gather_device = torch.device(device)
        self.strict_attn = strict_attn
        self.verbose = verbose
        self._printed = False

        cfg = self.base.config
        self.hidden_size = int(getattr(cfg, "hidden_size", 0))
        self.num_heads = int(getattr(cfg, "num_attention_heads", 0))
        self.moe_top_k = int(getattr(cfg, "num_experts_per_tok", getattr(cfg, "moe_top_k", 8)))

        # buffers
        self.layer_inputs = []
        self.attn_outputs = []
        self.expert_labels = []
        self.value_states = []

        self._register_hooks()

    # --------- utilities ----------
    def _input_device(self):
        # device_map="auto" 时，把 input_ids 放到第一个参数设备上
        return next(self.base.parameters()).device

    def _get_layers(self):
        # 适配常见 decoder layers 存放位置
        if hasattr(self.base, "model") and hasattr(self.base.model, "layers"):
            return self.base.model.layers
        # 某些结构可能是 transformer.h
        if hasattr(self.base, "transformer") and hasattr(self.base.transformer, "h"):
            return self.base.transformer.h
        raise RuntimeError("Cannot find decoder layers (tried base.model.layers and base.transformer.h)")

    def _reset_buffers(self, L: int):
        self.layer_inputs = [None] * L
        self.attn_outputs = [None] * L
        self.expert_labels = [None] * L
        self.value_states = [None] * L

    def _unwrap_attn_output(self, output):
        # 目标：找到形状像 (B,S,*) 的 tensor
        if torch.is_tensor(output):
            return output if output.dim() == 3 else None
        if isinstance(output, (tuple, list)):
            for item in output:
                if torch.is_tensor(item) and item.dim() == 3:
                    return item
        return None

    def _find_attn_module(self, layer: nn.Module):
        """
        Qwen3-Next 的注意力通常叫 linear_attn (Qwen3NextGatedDeltaNet)，
        投影层叫 in_proj_qkvz / out_proj，而不是 q_proj/k_proj/v_proj。
        """
        # 1) 先试最可靠的字段（Qwen3-Next）
        if hasattr(layer, "linear_attn") and getattr(layer, "linear_attn") is not None:
            return "linear_attn", getattr(layer, "linear_attn")

        # 2) 再试常见字段（其它 Qwen3 / 常规模型）
        for key in ["self_attn", "attn", "attention"]:
            m = getattr(layer, key, None)
            if m is not None:
                return key, m

        # 3) 兜底：递归找
        candidates = []
        for name, mod in layer.named_modules():
            if name == "":
                continue
            lname = name.lower()
            if not any(k in lname for k in ["linear_attn", "attn", "attention", "mha", "flash", "mixer"]):
                continue

            # ✅ Qwen3-Next 结构特征：in_proj_qkvz/out_proj
            has_in_qkvz = hasattr(mod, "in_proj_qkvz")
            has_out_proj = hasattr(mod, "out_proj") or hasattr(mod, "o_proj") or hasattr(mod, "out")

            # 兼容传统 attention：q/k/v/o proj
            has_qkv = (hasattr(mod, "q_proj") or hasattr(mod, "query_proj")) and \
                    (hasattr(mod, "k_proj") or hasattr(mod, "key_proj")) and \
                    (hasattr(mod, "v_proj") or hasattr(mod, "value_proj"))

            # 评分：优先 Qwen3-Next 的线性注意力模块
            score = 0
            if has_in_qkvz: score += 3
            if has_out_proj: score += 2
            if has_qkv: score += 3

            if score < 4:
                continue

            candidates.append((len(name), -score, name, mod))

        if not candidates:
            return None, None

        candidates.sort()
        _, _, name, mod = candidates[0]
        return name, mod

    def _find_v_proj(self, attn_mod: nn.Module):
        # 常见 v_proj 名称
        return getattr(attn_mod, "v_proj", None) or getattr(attn_mod, "value_proj", None)

    # --------- hook register ----------
    def _register_hooks(self):
        layers = self._get_layers()
        L = len(layers)

        # 绑定 layer_idx 的 hooks
        for layer_idx, layer in enumerate(layers):

            def make_pre_hook(idx):
                def pre_hook(module, inputs):
                    x = inputs[0]
                    # x: (B,S,H)
                    self.layer_inputs[idx] = x.detach().to(self.gather_device)
                return pre_hook

            layer.register_forward_pre_hook(make_pre_hook(layer_idx))

            attn_name, attn_mod = self._find_attn_module(layer)
            # if layer_idx == 0:
            #     print("\n[DEBUG] layer0 type =", type(layer))
            #     names = []
            #     for name, mod in layer.named_modules():
            #         lname = name.lower()
            #         if any(k in lname for k in ["attn", "attention", "mha", "flash", "mixer"]):
            #             # 打印模块类型和它是否含有常见投影
            #             names.append((
            #                 name,
            #                 mod.__class__.__name__,
            #                 int(hasattr(mod, "q_proj") or hasattr(mod, "query_proj")),
            #                 int(hasattr(mod, "k_proj") or hasattr(mod, "key_proj")),
            #                 int(hasattr(mod, "v_proj") or hasattr(mod, "value_proj")),
            #                 int(hasattr(mod, "o_proj") or hasattr(mod, "out_proj")),
            #             ))
            #     print("[DEBUG] attn-like modules in layer0 (name, cls, q,k,v,o flags):")
            #     for row in names[:80]:
            #         print("  ", row)
            #     print("[DEBUG] total candidates:", len(names), "\n")
            if attn_mod is None:
                msg = f"[wrapper] cannot find attn module in layer {layer_idx}"
                if self.strict_attn:
                    raise RuntimeError(msg)
                else:
                    if self.verbose and (not self._printed):
                        print(msg)
                    continue

            # hook: attn output
            def make_attn_hook(idx, name):
                def attn_hook(module, inputs, output):
                    out = self._unwrap_attn_output(output)
                    if out is None:
                        # 有些实现 output 不是 attn_output（可能返回别的结构），这里不强制报错
                        return
                    self.attn_outputs[idx] = out.detach().to(self.gather_device)
                    if self.verbose and (not self._printed) and idx < 2:
                        # 只打印前两层一次，避免刷屏
                        print(f"[wrapper] layer{idx} attn_mod='{name}' attn_out={tuple(out.shape)}")
                return attn_hook

            attn_mod.register_forward_hook(make_attn_hook(layer_idx, attn_name))

            # hook: value states（可选）
            v_proj = self._find_v_proj(attn_mod)
            if v_proj is not None:
                def make_v_hook(idx):
                    def v_hook(module, inputs, output):
                        v = output
                        if not (torch.is_tensor(v) and v.dim() == 3):
                            return
                        # v 可能是 (B,S,kv_heads*head_dim) 或 (B,S,H)
                        # 先尽量对齐到 hidden_size（如果就是 H 则直接收）
                        if v.size(-1) == self.hidden_size:
                            self.value_states[idx] = v.detach().to(self.gather_device)
                        else:
                            # 不做 GQA repeat（太依赖结构），先原样存，训练脚本别用 USE_AVG_VALUE
                            self.value_states[idx] = v.detach().to(self.gather_device)
                    return v_hook
                v_proj.register_forward_hook(make_v_hook(layer_idx))

        # 打印一次总层数
        if self.verbose:
            print(f"[wrapper] registered hooks for L={L} layers")
        self._printed = True

    # --------- forward ----------
    @torch.no_grad()
    def forward(
        self,
        input_ids,
        output_layer_input=False,
        output_attn_output=False,
        output_expert_label=False,
        output_value_states=False,
        **kwargs
    ):
        layers = self._get_layers()
        L = len(layers)
        self._reset_buffers(L)

        input_ids = input_ids.to(self._input_device())

        out = self.base(
            input_ids=input_ids,
            use_cache=False,
            output_router_logits=output_expert_label,
            return_dict=True,
            **kwargs,
        )

        if output_expert_label:
            B, S = input_ids.shape
            router_logits = getattr(out, "router_logits", None)
            if router_logits is None:
                router_logits = getattr(out, "moe_router_logits", None)
            if router_logits is None:
                raise RuntimeError("Cannot find router logits in model output (no router_logits/moe_router_logits)")

            labels = []
            for scores in router_logits:
                if scores.dim() == 2:
                    scores = scores.view(B, S, -1)
                idx = torch.topk(scores, k=self.moe_top_k, dim=-1).indices
                labels.append(idx.detach().to(self.gather_device))
            self.expert_labels = labels

        # 强校验：确保 attn_outputs 对齐（如果 strict_attn=True）
        if output_attn_output and self.strict_attn:
            missing = [i for i, x in enumerate(self.attn_outputs) if x is None]
            if missing:
                raise RuntimeError(f"Missing attn_output for layers: {missing[:20]} ... total_missing={len(missing)}")

        ret = {}
        if output_layer_input:
            ret["layer_input"] = self.layer_inputs
        if output_attn_output:
            ret["attn_output"] = self.attn_outputs
        if output_expert_label:
            ret["expert_label"] = self.expert_labels
        if output_value_states:
            ret["value_states"] = self.value_states
        return ret