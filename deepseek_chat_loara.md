# DeepSeek-7B-chat Lora 蒸馏微调

## 概述

本节我们简要介绍如何基于 transformers、peft 等框架，对 DeepSeek-7B-chat 模型进行 Lora 微调蒸馏，采用的是数据蒸馏方式，由大模型数据蒸馏得到。

这个教程会在同目录下给大家提供一个 [nodebook](./04-DeepSeek-7B-chat%20Lora%20微调.ipynb) 文件，来让大家更好的学习。

## 环境配置
### 主要依赖库说明

### 环境要求
- Python 3.8+
- CUDA 11.8+ (用于GPU训练)
- 至少24GB GPU内存 (推荐24GB+)
- 足够的磁盘空间存储模型和数据集
- GPU驱动和CUDA工具包已正确安装

在完成基本环境配置和本地模型部署的情况下，你还需要安装一些第三方库，可以使用以下命令：
#### 核心机器学习库
- **transformers**: Hugging Face的Transformers库，用于加载预训练模型和分词器
- **torch**: PyTorch深度学习框架，提供模型训练的基础功能
- **datasets**: Hugging Face的数据集处理库，用于高效处理训练数据
- **peft**: Parameter-Efficient Fine-Tuning库，提供LoRA等高效微调方法

#### 数据处理库
- **pandas**: 数据分析和处理库，用于读取JSON数据并转换格式
- **numpy**: 数值计算库（通过其他库间接使用）

#### 训练相关组件
- **AutoTokenizer**: 自动选择合适的分词器
- **AutoModelForCausalLM**: 自动加载因果语言模型
- **DataCollatorForSeq2Seq**: 序列到序列任务的数据整理器
- **TrainingArguments**: 训练参数配置类
- **Trainer**: Hugging Face的训练器，简化训练流程
- **GenerationConfig**: 文本生成配置类

```bash
pip install transformers==4.35.2
pip install peft==0.4.0
pip install datasets==2.10.1
pip install accelerate==0.20.3
pip install tiktoken
pip install transformers_stream_generator
```

在本节教程里，我们将微调蒸馏数据集放置在根目录 [/dataset]， 数据由json格式保存。

## 数据构造
训练一个模型的第一步是准备高质量的训练数据。在本次试验中，我们微调一个送祝福模型，对于一个送祝福模型，需要收集各种祝福语的数据，数据来源可以是公开的祝福语数据集、社交媒体、电子书籍或者任何包含丰富祝福语的文本；同时也可以通过专门prompt从LLM构建。
首先构建复杂的prompt，让llm生成需要的祝福：
```bash
scenes = ['生日', '春节', '元宵节', '端午节', '七夕节', '中秋节',
            '重阳节', '除夕', '腊八节','谈判顺利','乔迁新居', '周年纪念' ,'新婚快乐' ,'家庭和睦', '比赛取得好成绩' ,'发财','工作升职 ','康复', ]

styles = {
    "小红书":
    {
        "style_temple":"小红书风格，每条加入1-2个emoji表情包来增加趣味性。\n### 注意，你要参考下列句子的艺术风格进行祝福语撰写（注意！只看造句风格），祝福语结尾都带上语气助词词，参考句子为：{} ###",
        "if_example":True,
        "examples":
        [
    '默念你的名,祝你前途云蒸霞蔚，灿若星河。愿你度过的叫吉时，得到的叫如愿！',
    '希望你可以明确地爱，直接的厌恶，真诚的喜欢，站在太阳下的坦荡，大声无愧地称赞自己，学会爱自己！',
    '前方荣光万丈，身后温暖一方，凡是过往，皆为序章。',
    '愿所念之人 平安喜乐。愿所想之事 顺心如意！',
        ]
    },
    "正常":
    {
        "style_temple":"正常风格，有礼貌即可",
        "if_example":False,
        "examples":[]
    },
    "严肃":
    {
        "style_temple":"商业严肃风格，要求用在职场或长辈祝福上，显得有礼貌、干练,句子可以长一些",
        "if_example":False,
        "examples":[]
    }
}

random_finalprompt_sentence = [
    '', #默认情况
    '回答中可以不出现对象称谓和场景信息，也不用出现“愿你”“祝你”（对自己的长辈需要出现对象称谓和祝你），',
    '回答中可以不出现对象称谓和场景信息，',
    '回答中不用出现“愿你”“祝你”',
]

final_prompt = """
该祝福语字数小于 {} 字。 \n
请根据对象称谓及场景，写出符合对象的身份和场景气氛的祝福文案。要求的风格是：{} \n，注意不要有标题混在其中，对象称谓是：{}，祝福场景是：{}。 \n
{} 根据不同对象用不同的语气（尊敬、诙谐搞笑、亲近），请直接返回祝福文本，不要说任何其他话：
"""
```


## 数据清理
数据存在问题：
1. 句子长度返回错误，不是回答而是很短的一句
2. 加入语气助词后容易出现 `！啦~`

因此针对上述数据需要清理，把过于冗长，表达不完善，同时存在歧义和语法不通畅的清理。 同时把微调数据进行格式化：

```json
{
    "instrution":"回答以下用户问题，仅输出答案。",
    "input":"1+1等于几?",
    "output":"2"
}
```

其中，`instruction` 是用户指令，告知模型其需要完成的任务；`input` 是用户输入，是完成用户指令所必须的输入内容；`output` 是模型应该给出的输出。

即我们的核心训练目标是让模型具有理解并遵循用户指令的能力。因此，在指令集构建时，我们应针对我们的目标任务，针对性构建任务指令集。我们的目标是构建一个能够送祝福的个性化LLM，因此我们构造的指令形如：

```json
{
    "instruction": "你现在是一个送祝福大师，帮我针对不同人和事情、节日送对应的祝福",
    "input":"祝赵老师元宵节快乐,严肃风格",
    "output":"尊敬的赵老师，元宵佳节至，愿您福寿安康，智慧如灯。在这温馨的时刻，愿您家庭美满，幸福长存，如元宵般圆满甜蜜。谨祝元宵快乐，万事如意！"
}
```

我们所构造的全部指令数据集在dataset目录下。

## 数据格式化

`Lora` 训练的数据是需要经过格式化、编码之后再输入给模型进行训练的，如果是熟悉 `Pytorch` 模型训练流程的同学会知道，一般需要将输入文本编码为 input_ids，将输出文本编码为 `labels`，编码之后的结果都是多维的向量。
首先定义一个预处理函数，这个函数用于对每一个样本，编码其输入、输出文本并返回一个编码后的字典：

```python
def process_func(example):
    MAX_LENGTH = 384    # 分词器会将一个中文字切分为多个token，需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"User: {example['instruction']+example['input']}\n\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"Assistant: {example['output']}<｜end▁of▁sentence｜>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
```

这里的格式化输入参考了， [DeepSeek](https://github.com/deepseek-ai/DeepSeek-LLM) 官方github仓库中readme的指令介绍。

```text
User: {messages[0]['content']}

Assistant: {messages[1]['content']}<｜end▁of▁sentence｜>

User: {messages[2]['content']}

Assistant:
```

经过函数处理之后，数据token化之后的结果如下, 变成多维度的向量：
```
Input IDs: [5726, 25, 207, ... , 19304, 11423, 2220, 5088, 2160, 100001, 100001]
Labels : [-100, -100, -100, -100, ... , 2220, 5088, 2160, 100001, 100001]
```

## 加载tokenizer和半精度模型

模型以半精度形式加载，如果你的显卡比较新的话，可以用`torch.bfolat`形式加载。对于自定义的模型一定要指定`trust_remote_code`参数为`True`。

```python
tokenizer = AutoTokenizer.from_pretrained('./deepseek-ai/deepseek-llm-7b-chat/', use_fast=False, trust_remote_code=True)
tokenizer.padding_side = 'right' # 设置填充方向为右侧填充

model = AutoModelForCausalLM.from_pretrained('./deepseek-ai/deepseek-llm-7b-chat/', trust_remote_code=True, torch_dtype=torch.half, device_map="auto")
model.generation_config = GenerationConfig.from_pretrained('./deepseek-ai/deepseek-llm-7b-chat/')
model.generation_config.pad_token_id = model.generation_config.eos_token_id
```

## 定义LoraConfig

`LoraConfig`这个类中可以设置很多参数，主要关于lora结构设置，以下列出了主要修改的参数。

- `task_type`：模型类型
- `target_modules`：需要训练的模型层的名字，主要就是`attention`部分的层，不同的模型对应的层的名字不同。
- `r`：`lora`的秩，控制低秩分解的维度
- `lora_alpha`：`Lora alaph`，控制LoRA权重的缩放强度
- `lora_dropout`：`Lora`的dropout，防止过拟合
- `inference_mode`：推理模式，控制是否为推理

```python
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)
```

## 自定义 TrainingArguments 参数

`TrainingArguments`这个类的源码也介绍了每个参数的具体作用，当然大家可以来自行探索，这里就简单说几个常用的。

- `output_dir`: 模型和训练结果保存的目录
- `learning_rate`: 学习率，控制模型参数更新的步长
- `per_device_train_batch_size`: 每个设备的训练批次大小
- `num_train_epochs`: 训练的总轮数
- `weight_decay`: 权重衰减，用于防止过拟合
- `logging_steps`: 日志记录的步数间隔
- `save_steps`: 模型保存的步数间隔
- `gradient_checkpointing`: 是否启用梯度检查点，减少内存使用

```python
args = TrainingArguments(
    output_dir="./output/DeepSeek",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True
)
```

## 使用 Trainer 训练

```python
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
trainer.train()
```

## 模型推理

部署模型进行推理，测试结果。

```python
text = "祝姐姐生日快乐, 小红书风格"
inputs = tokenizer(f"User: {text}\n\n", return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

#### 推理结果：
输出结果符合平常祝福表达偏好，简短得体。
```text
User: 祝姐姐生日快乐, 小红书风格

Assistant: 愿岁月静好，笑靥如花🌸，幸福满溢，快乐无边🎉，姐，生日快乐呀！
```

#### 原始结果：
未微调蒸馏的原始模型，输出内容过于冗长，废话较多；表达内容不贴切！
```text
User: 祝姐姐生日快乐, 小红书风格

Assistant: 🎉🎂🎁🎈🎁🎂🎊🎈🎂🎉

我亲爱的姐姐，今天是你特别的一天，你的生日到了！在这个特别的日子里，我想对你说：祝你生日快乐！🎉🎂🎁🎈🎉

记得小时候，你总是会给我买好吃的糖果和玩具，还有那件我最喜欢的小红裙，让我感觉自己是世界上最幸福的小公主。你总是那么温柔、体贴，让我感到安心和温暖。👸🏻👸🏼👸🏽👸🏾👸🏿

现在，我也想给你送上一份小红书风格的生日祝福，希望你能感受到我的爱和祝福。💕

祝你每天都有好心情，身体健康，事业有成，家庭幸福，爱情甜蜜。🌹

最后，再次祝你生日快乐！🎉🎂🎁🎈🎉
```