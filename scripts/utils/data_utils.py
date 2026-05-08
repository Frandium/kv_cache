import os
import numpy as np
import torch
import json

from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import infer_auto_device_map


class DeepSeekDistillation(Dataset):
    def __init__(self, dataset_dir, max_seq_len, tokenizer, padding = True) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.files = sorted([dir for dir in os.listdir(dataset_dir) if dir.endswith('.txt')])
        self._load_file(0)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.combine_sequence = int(np.ceil(max_seq_len / 512))
        self.lines_per_file = len(self.file_texts) // self.combine_sequence
        self.padding = padding

    def __len__(self):
        return len(self.files) * self.lines_per_file

    def _load_file(self, file_idx):
        self.cur_file_idx = file_idx
        with open(f'{self.dataset_dir}/{self.files[self.cur_file_idx]}') as f:
            self.file_texts = f.readlines()

    def __getitem__(self, index):
        file_idx = index // self.lines_per_file
        if file_idx != self.cur_file_idx:
            self._load_file(file_idx)
        line_idx = index % self.lines_per_file
        if line_idx >= len(self.file_texts):
            print(f'{index}, {line_idx}, {len(self.file_texts)}, {self.dataset_dir}/{self.files[file_idx]}')
            line_idx = len(self.file_texts) - 1

        text = self.file_texts[line_idx * self.combine_sequence]
        for i in range(line_idx * self.combine_sequence + 1, line_idx * self.combine_sequence + self.combine_sequence):
            text += self.file_texts[i]

        if self.padding:
            token_ids = self.tokenizer(text, padding='max_length', 
                                   max_length = self.max_seq_len + 1, padding_side='right', 
                                   truncation=True, return_tensors='pt').input_ids
            real_len = len(self.tokenizer(text).input_ids)
        else:
            token_ids = self.tokenizer(text, return_tensors='pt').input_ids
            real_len = len(token_ids[0])

        return token_ids[0, :-1], token_ids[0, 1:], real_len


class TokenizedJSONLData(Dataset):
    def __init__(self, dataset_dir, max_seq_len, tokenizer, padding=True) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        # 递归收集所有子文件夹中的jsonl文件
        self.files = []
        for root, _, filenames in os.walk(dataset_dir):
            for filename in filenames:
                if filename.endswith('.txt'):
                    self.files.append(os.path.join(root, filename))
        self.files = sorted(self.files)
        # self.files = self.files[0*8800:0*8800+352]+self.files[1*8800:1*8800+210]+self.files[2*8800:2*8800+210]+self.files[3*8800:3*8800+210]
        
        self._load_file(0)  # 加载第一个文件
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.combine_sequence = 1 # int(np.ceil(max_seq_len / 1024))
        self.lines_per_file = len(self.file_texts) // self.combine_sequence
        self.padding = padding

    def __len__(self):
        return len(self.files) * self.lines_per_file

    def _load_file(self, file_idx):
        self.cur_file_idx = file_idx
        with open(self.files[self.cur_file_idx], 'r', encoding='utf-8') as f:
            # 直接读取每行作为文本，去除首尾空白字符
            self.file_texts = [line.strip() for line in f if line.strip()]
            # 过滤掉空行

    def __getitem__(self, index):
        file_idx = index // self.lines_per_file
        if file_idx != self.cur_file_idx:
            self._load_file(file_idx)
        
        # 计算行索引并确保不越界
        line_idx = index % self.lines_per_file
        max_line_idx = len(self.file_texts) // self.combine_sequence - 1
        line_idx = min(line_idx, max_line_idx)

        # 组合多个文本片段
        # start_idx = line_idx * self.combine_sequence
        # end_idx = start_idx + self.combine_sequence
        text = json.loads(self.file_texts[line_idx]) # ''.join(self.file_texts[start_idx:end_idx])

        # 处理tokenization
        if self.padding:
            token_ids = self.tokenizer(
                text, 
                padding='max_length',
                max_length=self.max_seq_len + 1,
                padding_side='right',
                truncation=True,
                return_tensors='pt'
            ).input_ids
            real_len = len(self.tokenizer(text).input_ids)
        else:
            token_ids = self.tokenizer(text, return_tensors='pt').input_ids
            real_len = len(token_ids[0])
        
        return token_ids[0, :-1], token_ids[0, 1:], real_len


class pruned_Data(Dataset):
    def __init__(self, dataset_dir, max_seq_len, tokenizer, D_PER, padding=True) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        # 递归收集所有子文件夹中的jsonl文件
        self.files = []
        for root, _, filenames in os.walk(dataset_dir):
            if "G1L0" in root or "G1L1" in root or "G1L2" in root:
            # if "_0" in root or "_1" in root or "_2" in root:
                for filename in filenames:
                    if filename.endswith('.txt'):
                        self.files.append(os.path.join(root, filename))
            else:
                continue
        self.files = sorted(self.files, key=lambda x: os.path.basename(x))
        if D_PER >= 1:
            self.files = self.files
        else:
            self.files = self.progressive_prune(self.files, D_PER)
        
        self._load_file(0)  # 加载第一个文件
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.combine_sequence = 1 # int(np.ceil(max_seq_len / 1024))
        self.lines_per_file = len(self.file_texts) // self.combine_sequence
        self.padding = padding

    def __len__(self):
        return len(self.files) * self.lines_per_file

    def _load_file(self, file_idx):
        self.cur_file_idx = file_idx
        with open(self.files[self.cur_file_idx], 'r', encoding='utf-8') as f:
            # 直接读取每行作为文本，去除首尾空白字符
            self.file_texts = [line.strip() for line in f if line.strip()]
            # 过滤掉空行

    def __getitem__(self, index):
        file_idx = index // self.lines_per_file
        if file_idx != self.cur_file_idx:
            self._load_file(file_idx)
        
        # 计算行索引并确保不越界
        line_idx = index % self.lines_per_file
        max_line_idx = len(self.file_texts) // self.combine_sequence - 1
        line_idx = min(line_idx, max_line_idx)

        # 组合多个文本片段
        # start_idx = line_idx * self.combine_sequence
        # end_idx = start_idx + self.combine_sequence
        text = json.loads(self.file_texts[line_idx]) # ''.join(self.file_texts[start_idx:end_idx])

        # 处理tokenization
        if self.padding:
            token_ids = self.tokenizer(
                text, 
                padding='max_length',
                max_length=self.max_seq_len + 1,
                padding_side='right',
                truncation=True,
                return_tensors='pt'
            ).input_ids
            real_len = len(self.tokenizer(text).input_ids)
        else:
            token_ids = self.tokenizer(text, return_tensors='pt').input_ids
            real_len = len(token_ids[0])
        
        return token_ids[0, :-1], token_ids[0, 1:], real_len
    def progressive_prune(self, sorted_list, D_PER, num_bins=20):
        """
        对排序好的列表进行分段递减删减
        
        Args:
            sorted_list (list): 已排序列表
            D_PER (float): 最终保留比例 (0, 1]
            num_bins (int): 分段数，默认 20
        
        Returns:
            list: 删减后的列表
        """
        # print("D_PER:", D_PER)
        assert 0 < D_PER <= 1
        N = len(sorted_list)
        if N == 0:
            return []

        # 最大删减比例（第一段）
        r_max = min(1.0, 2 * (1 - D_PER))

        # 分段
        bins = []
        base = N // num_bins
        extra = N % num_bins

        idx = 0
        for i in range(num_bins):
            size = base + (1 if i < extra else 0)
            bins.append(sorted_list[idx: idx + size])
            idx += size

        kept = []

        for i, bin_items in enumerate(bins):
            if not bin_items:
                continue

            # 第 i 段删减比例（线性递减）
            r_i = r_max * (1 - i / (num_bins - 1))
            keep_ratio = 1 - r_i

            keep_n = max(0, int(round(len(bin_items) * keep_ratio)))

            # 从该段中“均匀保留”
            if keep_n >= len(bin_items):
                kept.extend(bin_items)
            elif keep_n > 0:
                step = len(bin_items) / keep_n
                indices = [int(j * step) for j in range(keep_n)]
                kept.extend(bin_items[j] for j in indices)

        # 防止浮点误差，最终对齐目标数量
        target_n = int(round(N * D_PER))
        if len(kept) > target_n:
            kept = kept[:target_n]
        elif len(kept) < target_n:
            # 极端情况下补尾部
            missing = target_n - len(kept)
            kept.extend(sorted_list[-missing:])

        return kept




import pyarrow.parquet as pq

class ParquetProblemDataset(Dataset):
    """
    从 Parquet 文件加载问题数据并进行 tokenization 的 PyTorch Dataset。
    不使用 pandas。

    期望的 Parquet Schema:
        problem: string
        level: string
        type: string
        solution: string

    返回字典格式:
        {
         'input_ids': Tensor,
         'attention_mask': Tensor,
         'labels': Tensor (tokenized solution),
         'problem_text': str (原始问题文本), # 可选，方便调试
         'solution_text': str (原始答案文本) # 可选，方便调试
        }
    """

    def __init__(self, parquet_file_path, max_length,  tokenizer, include_texts_for_debug=False):
        """
        初始化 Dataset。

        Args:
            parquet_file_path (str): Parquet 文件路径。
            tokenizer: Hugging Face Tokenizer 实例。
            max_length (int, optional): 序列最大长度。如果为 None，则由 tokenizer 决定或使用 'longest'。
                                       通常训练/验证时会设定一个固定值。
            include_texts_for_debug (bool): 是否在返回的样本中包含原始文本。
                                           默认 False，用于生产环境以节省内存。
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_texts = include_texts_for_debug

        # 读取 Parquet 文件
        try:
            table = pq.read_table(parquet_file_path)
            print(f"成功加载 Parquet 文件: {parquet_file_path}")
        except Exception as e:
            raise RuntimeError(f"无法读取 Parquet 文件 {parquet_file_path}: {e}")

        # --- 关键修改：不使用 pandas 提取列数据 ---
        # 获取 PyArrow Array 对象
        try:
            problem_array = table['problem']
            level_array = table['level']
            type_array = table['type']
            solution_array = table['solution']
        except KeyError as e:
            raise KeyError(f"Parquet 文件缺少预期的列: {e}")

        # 将 PyArrow Array 转换为 Python 列表
        # .to_pylist() 是 PyArrow 提供的将数组内容转换为 Python 列表的方法
        self.problems = problem_array.to_pylist()
        self.levels = level_array.to_pylist()
        self.types = type_array.to_pylist()
        self.solutions = solution_array.to_pylist()
        # --- 修改结束 ---

        if not (len(self.problems) == len(self.levels) == len(self.types) == len(self.solutions)):
            raise ValueError("Parquet 文件中的列长度不一致!")

        self.len = len(self.problems)
        print(f"Dataset 已初始化，共包含 {self.len} 个样本。")

    def __len__(self):
        """返回数据集大小。"""
        return self.len

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.len:
             raise IndexError(f"Index {idx} is out of bounds for dataset of size {self.len}")

        # 获取原始文本
        problem_text = self.problems[idx]
        level_text = self.levels[idx]
        type_text = self.types[idx]
        solution_text = self.solutions[idx]

        # 构造 Prompt (根据你的具体需求调整这里的格式)
        # 示例格式: "Problem: {problem}\nLevel: {level}\nType: {type}\nSolution:"
        prompt = f"Problem: {problem_text}"

        # --- Tokenization ---
        # Tokenize Prompt (输入)
        encoding_inputs = self.tokenizer(
            prompt,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length' if self.max_length else 'longest',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize Solution (标签)
        encoding_labels = self.tokenizer(
            solution_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length' if self.max_length else 'longest',
            truncation=True,
            return_tensors='pt'
        )

        try:
            level = int(level_text[-1])
        except ValueError:
            level = 0
        # 准备返回的样本字典
        sample = {
            'input_ids': encoding_inputs['input_ids'].squeeze(),       # [1, seq_len] -> [seq_len]
            'attention_mask': encoding_inputs['attention_mask'].squeeze(),
            'labels': encoding_labels['input_ids'].squeeze(),          # [1, seq_len] -> [seq_len]
            'level': level,
        }

        # 可选：包含原始文本以便调试
        if self.include_texts:
            sample['problem_text'] = problem_text
            sample['solution_text'] = solution_text
            sample['prompt_text'] = prompt # 也可以加上构造的 prompt

        return sample
