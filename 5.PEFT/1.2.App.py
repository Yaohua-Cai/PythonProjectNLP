# ============================ #
#   用 4bit LoRA 适配器做推理    #
#   模型: Qwen3-1.7B (本地)     #
# ============================ #

# —— 1) 在导入 transformers 之前禁用 TF/JAX，避免 tf_keras 报警 —— #
import os                                               # 操作系统接口
os.environ["TRANSFORMERS_NO_TF"] = "1"                 # 禁用 TensorFlow 相关路径（我们只用 PyTorch）
os.environ["TRANSFORMERS_NO_JAX"] = "1"                # 禁用 JAX 相关路径（保持纯 PyTorch）

# —— 2) 常用依赖导入 —— #
import torch                                            # PyTorch 张量/设备管理
from transformers import (                              # Transformers 生态
    AutoTokenizer, AutoModelForCausalLM,               # 分词器与自回归模型加载器
    BitsAndBytesConfig                                 # bitsandbytes 的 4bit 量化配置
)
from peft import PeftModel                              # PEFT 推理时把 LoRA 适配器挂到基座模型上

# —— 3) 路径与基础配置 —— #
MODEL_ID   = "../../model/Qwen3-1.7B"                               # 你的“基座”模型（本地目录或已安装的模型名）
# ADAPTER_DIR = "outputs/qwen3_lora"              # 你的 LoRA 输出目录（请改成你的真实 output 路径）
ADAPTER_DIR = "outputs/qwen3_adalora"              # 你的 LoRA 输出目录（请改成你的真实 output 路径）
MAX_NEW_TOKENS = 300                                     # 生成的最大新 token 数（按你要求设为 30）
USE_BEAM = True                                         # 推理策略：True=beam search（更稳），False=采样（更发散）

# —— 4) 加载分词器（确保有 pad_token） —— #
tokenizer = AutoTokenizer.from_pretrained(              # 从基座模型目录加载分词器
    MODEL_ID,                                           # 基座模型（和训练的一致）
    trust_remote_code=True,                             # 允许模型自定义代码
    use_fast=True                                       # 使用 fast 分词器（更快）
)
if tokenizer.pad_token is None:                         # 自回归模型常见：pad_token 可能未定义
    tokenizer.pad_token = tokenizer.eos_token           # 将 pad_token 对齐到 eos_token（训练/推理更稳）

# —— 5) 4bit 量化配置（必须与训练时一致：QLoRA） —— #
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,                                  # 加载为 4bit 权重
    bnb_4bit_compute_dtype=torch.bfloat16,              # 前向计算用 bfloat16（稳定且高效）
    bnb_4bit_use_double_quant=True,                     # 双重量化（进一步省显存）
    bnb_4bit_quant_type="nf4",                          # NF4 量化类型（QLoRA 推荐）
)

# —— 6) 加载“4bit 基座模型” —— #
base_model = AutoModelForCausalLM.from_pretrained(      # 从基座目录加载模型权重
    MODEL_ID,                                           # 与训练一致的基座
    trust_remote_code=True,                             # 允许自定义代码
    device_map="auto",                                  # 自动把权重分配到可用 GPU/CPU
    quantization_config=bnb_cfg                         # ★ 传入 4bit 配置（保持与训练一致）
)

# —— 7) 把 LoRA 适配器挂载到基座模型上（推理只需这一步，不要 merge） —— #
model = PeftModel.from_pretrained(                      # 用 PEFT 把 LoRA 适配器加载进来
    base_model,                                         # 基座模型（4bit）
    ADAPTER_DIR                                         # ★ 你的 LoRA 输出目录
).eval()                                                # 切换为 eval 模式（推理更稳、更省显存）

# —— 8) （可选）打印一次 4bit/LoRA 是否生效的检查信息 —— #
model.print_trainable_parameters()                      # 理论上应只显示很少的可训练参数（但此处是推理，纯检查）
first_linear = None                                     # 准备一个变量存第一个 4bit 线性层名称
for name, module in model.named_modules():              # 遍历模块
    cname = module.__class__.__name__.lower()           # 模块类名小写
    if "linear4bit" in cname:                           # 若包含 linear4bit，则说明 4bit 成功
        first_linear = (name, module.__class__.__name__)# 存下第一个匹配
        break                                           # 记录一个就够
print("first 4bit module:", first_linear)               # 打印看下（None 表示没匹配到，需检查环境）

# —— 9) 构造对话消息（使用 chat_template，避免提示回显/格式错乱） —— #
messages = [                                            # 准备一轮简短对话
    {"role": "system", "content": "你是一个乐于助人的助手。请直接给出最终答案，不要展示思考过程。"},  # 系统提示，约束风格
    {"role": "user",   "content": "用通俗的话解释 attention mask 的作用。"},                           # 用户问题（可替换）
]

# —— 10) 用 chat_template 组装提示（add_generation_prompt=True 让模型进入“生成助手回复”模式） —— #
prompt_text = tokenizer.apply_chat_template(            # 让分词器按模型的对话模板拼接文本
    messages,                                           # 上面的消息列表
    tokenize=False,                                     # 返回字符串（而不是 token id）
    add_generation_prompt=True                          # 添加“助手开始生成”的标记
)

# —— 11) 把提示转为张量并放到模型设备上 —— #
inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)  # 分词 → 张量 → 放到相同设备

# —— 12) 收集多个可能的 eos（不同 Qwen 变体可能用 <|im_end|> / <|endoftext|>） —— #
def collect_eos_ids(tok):                               # 定义一个工具函数收集 eos id 列表
    cands = []                                          # 候选列表
    for t in ["<|im_end|>", "<|endoftext|>"]:           # 常见特殊结束符
        try:
            tid = tok.convert_tokens_to_ids(t)          # 把 token 文本转为 id
            if tid is not None and tid != tok.unk_token_id:
                cands.append(tid)                       # 合法则加入
        except Exception:
            pass                                        # 有的模型可能没有这个特殊符号
    if tok.eos_token_id is not None:                    # 把通用 eos 也加入
        cands.append(tok.eos_token_id)
    uniq = []                                           # 去重并保持顺序
    [uniq.append(x) for x in cands if x not in uniq]
    return uniq or None                                 # 如果为空就返回 None

eos_ids = collect_eos_ids(tokenizer)                    # 得到一个 eos id 列表（可直接传给 generate）

# —— 13) （可选）屏蔽“思维标记”之类的 token（如果它们恰好是单 token） —— #
bad_tokens = ["<think>", "</think>", "<|current_state|>"]  # 可能不希望模型输出的标记
bad_ids = []                                              # 存放单 token 的禁止词 id
for t in bad_tokens:                                     # 遍历每个标记
    ids = tokenizer.encode(t, add_special_tokens=False)  # 编码成 id 序列
    if len(ids) == 1:                                    # 只有“恰好是 1 个 token”时才能屏蔽
        bad_ids.append(ids)                              # bad_words_ids 要求形如 [[id1],[id2],...]
bad_words_ids = bad_ids if bad_ids else None             # 若都是多 token，就不传（靠 system 提示约束）

# —— 14) 选择生成策略：Beam Search（稳）或 采样（更有创造性） —— #
gen_kwargs = dict(                                       # 准备一个生成参数字典
    max_new_tokens=MAX_NEW_TOKENS,                       #  新生成 token 上限
    repetition_penalty=1.05,                             # 轻微惩罚重复
    eos_token_id=eos_ids,                                # 在任一 eos 处停止
    bad_words_ids=bad_words_ids                          # 尝试屏蔽“思维标记”
)
if USE_BEAM:                                             # 如果选择 beam search
    gen_kwargs.update(dict(                              # 更新为 Beam 配置
        do_sample=False,                                 # beam 一般不采样
        num_beams=5,                                     # beam=5（可按需调大）
        length_penalty=1.0                               # 长度惩罚，1.0 为中性
    ))
else:                                                    # 否则使用采样策略
    gen_kwargs.update(dict(                              # 更新为采样配置
        do_sample=True,                                  # 开启采样
        top_p=0.9,                                       # nucleus sampling
        temperature=0.7                                  # 温度（越大越发散）
    ))

# —— 15) 生成，并仅解码“新增 token”（避免回显提示） —— #
with torch.no_grad():                                    # 推理阶段不需要梯度
    outputs = model.generate(                            # 调用 generate 进行自回归生成
        **inputs,                                        # 输入张量（包含 prompt）
        **gen_kwargs                                     # 上面准备的生成参数
    )
new_tokens = outputs[0, inputs["input_ids"].shape[1]:]   # 取从 prompt 之后的新生成部分
result_text = tokenizer.decode(                          # 把新 token 解码为字符串
    new_tokens, skip_special_tokens=True                 # 忽略特殊符号
).strip()                                                # 去掉两端空白
print(result_text)                                       # 打印最终结果（只包含模型回复）
