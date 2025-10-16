# ============================ #
#   Qwen3-1.7B LoRA 训练+推理   #
#   版本：trl=0.22.2 / tfm=4.55.2/ peft==0.17.1
#   要点：20条messages，输出30 token  #
# ============================ #

# —— 1) 在 transformers 导入前禁用 TF/JAX，避免 tf_keras 警告 —— #
import os                                              # 操作系统接口（设置环境变量）
os.environ["TRANSFORMERS_NO_TF"] = "1"                # 禁用 TensorFlow 路径（不加载 TF 相关）
os.environ["TRANSFORMERS_NO_JAX"] = "1"               # 禁用 JAX 路径（保持纯 PyTorch）

# —— 2) 常规依赖 —— #
import json                                            # 读取本地 JSONL 数据用
import torch                                           # PyTorch
from datasets import Dataset                           # 轻量构造训练数据
from transformers import (                             # HuggingFace Transformers 组件
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
from trl import SFTTrainer, SFTConfig                  # TRL 的 SFT 训练器与配置
from peft import LoraConfig, TaskType, PeftModel       # PEFT：LoRA 配置与推理时加载适配器

# —— 3) 可改参数（根据你的机器与需求调整） —— #
MODEL_ID   = "../../model/Qwen3-1.7B"                              # 你的本地模型目录/名称
OUTPUT_DIR = "outputs/qwen3_lora"          # 训练输出目录（LoRA 适配器会保存在这里）
MAX_LEN    = 1024                                       # 训练与tokenize的最大序列长度
EPOCHS     = 60                                          # 训练轮次（示例设为2；真实训练可更大）
BATCH_SIZE = 4                                          # 每设备 batch（显存不够就减小）
GRAD_ACC   = 8                                          # 梯度累积步数（等效放大batch）
LR         = 2e-4                                       # 学习率（LoRA常用范围 1e-4~5e-4）
WARMUP     = 0.03                                       # warmup 比例（3%）
WEIGHT_DECAY = 0.1                                      # 权重衰减（L2正则）
MAX_GRAD_NORM = 1.0                                     # 梯度裁剪上限
LOAD_4BIT  = True                                      # 是否启用4bit量化（QLoRA）。若未安装bitsandbytes请设为False
USE_FLASH  = False                                      # 若未安装 flash-attn 请保持 False
REPEAT_K   = 200                                        # 小样本时重复次数（仅为“能看到下降”，真实训练请用大语料并设为1）

# —— 4) 构造一个最小训练数据（你可以改成读取 data/train.jsonl） —— #
#    JSONL 每行形如：{"instruction":"...", "output":"..."}
JSONL_PATH = "data/train.jsonl"                         # 如有更大数据集，放在这个路径
def load_training_samples():
    """优先从 JSONL 读取；否则回退到两条内置样本（仅用于跑通流程）"""
    if os.path.isfile(JSONL_PATH):                      # 若 JSONL 存在
        rows = []
        with open(JSONL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)                  # 解析为字典
                if "instruction" in obj and "output" in obj:
                    rows.append({"instruction": obj["instruction"], "output": obj["output"]})
        if len(rows) > 0:
            return rows                                 # 返回读取到的样本

    return [
        {"instruction": "把下面句子翻成英文：我爱自然语言处理。", "output": "I love natural language processing."},
        {"instruction": "用一句话解释注意力机制。",           "output": "根据上下文给不同词分配权重，从而聚焦关键信息。"},
        {"instruction": "把下面句子翻成英文：我爱自然语言处理。", "output": "I love natural language processing."},
        {"instruction": "用一句话解释注意力机制。", "output": "根据上下文给不同词分配权重，从而聚焦关键信息。"},
        {
            "instruction": "将这段话压缩成一句话：近年来，大语言模型在对话、翻译和编程等任务上表现突出，但也带来了安全与偏见问题，需要负责任地使用。",
            "output": "大语言模型能力强但伴随安全与偏见风险，需要负责任地应用。"},
        {"instruction": "判断情感（正面或负面）：这个产品太让人失望了。", "output": "负面"},
        {"instruction": "提取关键词：深度学习模型需要大量数据和计算资源来训练。",
         "output": "深度学习, 模型, 数据, 计算资源, 训练"},
        {"instruction": "把这句话改写得更礼貌：把报告立刻给我。", "output": "请尽快把报告发给我，谢谢。"},
        {"instruction": "纠正语法并保持原意：He go to school every day.", "output": "He goes to school every day."},
        {"instruction": "把这句话翻成中文：Transformers are powerful sequence models.",
         "output": "Transformer 是一种强大的序列模型。"},
        {"instruction": "为这段文字生成一个简短标题：人工智能正被用于医疗影像分析、客服自动化和交通优化等领域。",
         "output": "人工智能应用概览"},
        {"instruction": "用一句话解释 Transformer 的核心思想。", "output": "通过自注意力机制建模长程依赖，并行处理序列。"},
        {"instruction": "给出三条学习自注意力的建议。",
         "output": "从点积注意力公式入门；用小例子手算一遍；可视化权重理解关注点。"},
        {"instruction": "识别这句话的语言：Bonjour, comment ça va ?", "output": "法语"},
        {"instruction": "续写一句，使语义合理：春天到了，花园里", "output": "开满了五彩的花，空气里带着清新的泥土香。"},
        {"instruction": "将主动语态改为被动语态：The committee approved the proposal.",
         "output": "The proposal was approved by the committee."},
        {"instruction": "从句子中抽取时间和地点：我们将于2024年6月18日在北京召开发布会。",
         "output": "时间: 2024年6月18日; 地点: 北京"},
        {"instruction": "写一个匹配电子邮箱的正则表达式。", "output": "^[\\w\\.-]+@[\\w\\.-]+\\.[a-zA-Z]{2,}$"},
        {"instruction": "把下面三点合并为条目式摘要：1) 提升数据质量；2) 调整学习率；3) 增大批量大小。",
         "output": "- 提升数据质量\\n- 调整学习率\\n- 增大批量大小"},
        {"instruction": "给出单词 fast 的三个反义词。", "output": "slow, sluggish, leisurely"},
        {"instruction": "把这句话改写得更幽默：今天加班到很晚。", "output": "今天和公司谈了场持久的恋爱，直到深夜才分手。"},
        {"instruction": "用通俗的话解释 attention mask 的作用。",
         "output": "用来指定哪些位置可见、哪些位置要遮住，避免模型看见不该看的信息。"}

    ]

# —— 5) 加载分词器（注意自回归LM常把 pad_token 设为 eos_token） —— #
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,                                           # 模型目录/名称
    trust_remote_code=True,                             # 允许自定义代码
    use_fast=True                                       # 使用更快的分词器
)
if tokenizer.pad_token is None:                         # 如果没有 pad_token
    tokenizer.pad_token = tokenizer.eos_token           # 用 eos 作为 pad（常见做法）

# —— 6) 把样本映射成单列 "text"，用 chat_template 组织对话 —— #
def to_text_row(example: dict) -> dict:
    """
    输入：{"instruction": "...", "output": "..."}
    输出：{"text": 用 chat_template 组织好的训练文本}
    训练时我们把 system/user/assistant 拼在一起，目标就是最后的 assistant 内容。
    """
    messages = [
        {"role": "system",    "content": "你是一个乐于助人的助手。"},
        {"role": "user",      "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    text = tokenizer.apply_chat_template(               # 使用模型自带的聊天模板
        messages,
        tokenize=False,                                 # 返回文本串（不分词）
        add_generation_prompt=False                     # 不额外添加“等待生成”的提示
    )
    return {"text": text}                               # SFTTrainer 需要 dataset_text_field="text"

# —— 7) 构建训练数据集（小样本时重复 REPEAT_K 次，便于看到 loss 下降） —— #
raw_samples = load_training_samples()                   # 载入原始样本（JSONL 或内置）
if len(raw_samples) <= 5:                               # 如果样本很少
    raw_samples = raw_samples * REPEAT_K                # 重复样本（仅演示；真实训练请删除此行）
train_ds = Dataset.from_list(raw_samples).map(          # 构造 HuggingFace Dataset 并映射
    to_text_row, remove_columns=list(raw_samples[0].keys())
)

# —— 8) （可选）4bit 量化配置（QLoRA）。若未安装 bitsandbytes 请保持 False —— #
bnb_cfg = None                                          # 默认不量化
if LOAD_4BIT:                                           # 若要节省显存并已安装 bnb
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,                              # 加载为 4bit 权重
        bnb_4bit_compute_dtype=torch.bfloat16,          # 计算用 bfloat16（更稳）
        bnb_4bit_use_double_quant=True,                 # 双重量化进一步省显存
        bnb_4bit_quant_type="nf4",                      # NF4 量化方案
    )

# —— 9) LoRA 配置（覆盖注意力和FFN投影层） —— #
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,                       # 自回归语言建模
    r=16,                                               # LoRA 秩（8/16/32 可调）
    lora_alpha=32,                                      # 放大系数
    lora_dropout=0.05,                                  # LoRA dropout
    bias="none",                                        # 不训练原始 bias
    target_modules=[                                    # Qwen 常见模块命名
        "q_proj","k_proj","v_proj","o_proj",            # 注意力四投影
        "gate_proj","up_proj","down_proj"               # FFN 三投影
    ]
)

# —— 10) SFT 训练配置（注意 trl=0.22.2 的参数名） —— #
sft_args = SFTConfig(
    output_dir=OUTPUT_DIR,                              # 输出目录
    num_train_epochs=EPOCHS,                            # 训练轮数
    per_device_train_batch_size=BATCH_SIZE,             # 每设备 batch
    gradient_accumulation_steps=GRAD_ACC,               # 梯度累积
    learning_rate=LR,                                   # 学习率
    lr_scheduler_type="cosine",                         # 余弦退火
    warmup_ratio=WARMUP,                                # warmup 比例
    weight_decay=WEIGHT_DECAY,                          # 权重衰减
    max_grad_norm=MAX_GRAD_NORM,                        # 梯度裁剪
    logging_steps=10,                                   # 日志步频
    save_steps=200,                                     # 保存步频（真实训练可调大）
    save_total_limit=3,                                 # 最多保留几个 checkpoint
    bf16=torch.cuda.is_available(),                     # GPU 上用 bf16
    gradient_checkpointing=True,                        # 检查点以省显存
    dataset_text_field="text",                          # 数据列名
    max_length=MAX_LEN,                                 # 关键：本版本用 max_length
    packing=False if not USE_FLASH else True,           # 未装 flash-attn 时务必 False
    optim="adamw_torch",                                # 通用 AdamW
    model_init_kwargs={                                 # 透传给 from_pretrained 的参数
        "quantization_config": bnb_cfg,                 # 4bit 配置（None即不量化）
        "device_map": "auto",                           # 自动分配设备
        # 若已安装 flash-attn，且想开启 packing，可解开下行并置 USE_FLASH=True：
        # "attn_implementation": "flash_attention_2",
    },
)

# —— 11) 构建 SFTTrainer（由 Trainer 加载基座并注入 LoRA） —— #
trainer = SFTTrainer(
    model=MODEL_ID,                                     # 直接传模型目录/名称，内部 from_pretrained
    args=sft_args,                                      # 训练参数
    train_dataset=train_ds,                             # 训练数据
    processing_class=tokenizer,                         # v0.22.2 用 processing_class 传入 tokenizer
    peft_config=lora_cfg,                               # 让 Trainer 自动包裹 LoRA
)

# —— 12) 打印可训练参数占比（通常 ~1%） —— #
trainer.model.print_trainable_parameters()              # 验证 LoRA 是否生效

# —— 13) 开始训练 —— #
trainer.train()                                         # 跑若干轮（小数据会很快）

# —— 14) 保存 LoRA 适配器与分词器 —— #
trainer.model.save_pretrained(OUTPUT_DIR)               # 保存 LoRA 适配器权重
tokenizer.save_pretrained(OUTPUT_DIR)                   # 保存 tokenizer（便于复现/部署）

# ===================== 推理/验证（构造20条messages，输出30 token） ===================== #

# —— 15) 载入“基座 + LoRA 适配器” —— #
base = AutoModelForCausalLM.from_pretrained(            # 加载基础模型
    MODEL_ID, trust_remote_code=True, device_map="auto",
    quantization_config=bnb_cfg                         # 若训练时用了 QLoRA，这里也保持一致
)
model = PeftModel.from_pretrained(base, OUTPUT_DIR).eval()  # 加载 LoRA；eval 模式推理更稳

# —— 16) 收集可能的 EOS（兼容不同模板的停止符） —— #
def collect_eos_ids(tok):
    """收集多个可能的 EOS id，供 generate 停止使用。"""
    cands = []
    for t in ["<|im_end|>", "<|endoftext|>"]:           # 常见特殊结束符
        try:
            tid = tok.convert_tokens_to_ids(t)
            if tid is not None and tid != tok.unk_token_id:
                cands.append(tid)
        except Exception:
            pass
    if tok.eos_token_id is not None:                    # 再把通用 eos 也加入
        cands.append(tok.eos_token_id)
    uniq = []
    [uniq.append(x) for x in cands if x not in uniq]    # 去重保持顺序
    return uniq or None

eos_ids = collect_eos_ids(tokenizer)                    # 生成时使用

# —— 17) 构造“20 条 messages”的对话上下文 —— #
# 规则：1条 system + 18条历史（9对 user/assistant）+ 最后一条 user 询问 = 20条
messages = []                                           # 准备一个空的消息列表
messages.append({"role": "system", "content": "你是一个乐于助人的助手。请直接给出最终答案，不要展示思考过程。"})  # 第1条：system


# 最后一条第20条：当前用户的真实提问（你可以换成你的任务）
messages.append({"role": "user", "content": "给出单词 fast 的三个反义词。"})                   # 第20条：当前 user

# —— 18) 用 chat_template 拼成提示，开启 add_generation_prompt=True 让模型进入“生成助手回复”状态 —— #
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,                                     # 返回纯文本
    add_generation_prompt=True                          # 添加“assistant:” 起始以便生成
)

# —— 19) 把提示转成张量并放到模型设备（如你的 5070Ti） —— #
inputs = tokenizer([prompt], return_tensors="pt").to(model.device)  # 分词并移动到对应设备

# —— 20) 屏蔽可能的“思维标记”（若它们刚好是单 token；多 token 场景主要靠 system 提示约束） —— #
bad_tokens = ["<think>", "</think>", "<|current_state|>"]           # 常见“思维/状态”标记
bad_ids = []
for t in bad_tokens:
    ids = tokenizer.encode(t, add_special_tokens=False)              # 把标记编码成 token id 序列
    if len(ids) == 1:                                               # 仅当恰好是单 token，才能屏蔽
        bad_ids.append(ids)
bad_words_ids = bad_ids if bad_ids else None                        # 若都是多 token，就不传（system 已约束）

# —— 21) 生成配置：输出上限 30 个 token（按你的要求），使用 beam search 更稳 —— #
with torch.no_grad():                                               # 推理无需梯度
    gen = model.generate(
        **inputs,                                                   # 输入张量
        max_new_tokens=100,                                          # 输出上限=30 token
        do_sample=False,                                            # 关闭采样（可控性更强）
        num_beams=5,                                                # beam search = 5
        length_penalty=1.0,                                         # 长度惩罚（中性）
        repetition_penalty=1.05,                                    # 轻微惩罚重复
        eos_token_id=eos_ids,                                       # 任意 EOS 均可提前停止
        bad_words_ids=bad_words_ids                                 # 尝试屏蔽“思维标记”
    )

# —— 22) 只解码“新增 token”（避免把提示回显），并打印结果 —— #
new_tokens = gen[0, inputs["input_ids"].shape[1]:]                  # 只取生成段
result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()  # 解码为字符串
print(result)                                                       # 打印最终答案
