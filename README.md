# openai-library-tutorial
openai库的简单教程。A samply guide to using the openai library.

# OpenAI API使用说明

## 一、基本概念

**Token(符号)**
模型通过将文本分解成token来理解和处理。token可以是单词，也可以是大块的字符。例如，hamburger” gets broken up into the tokens这个词可以分解为 “ham”, “bur” ,“ger”这几个token。

**Prompt(提示)**

通俗的来说，就是对语言模型的输入，模型的回答即对prompt的回应(response),通过对prompt的设计可以使模型产生稳定的符合需求的输出。

## 二、API Key 配置

​      一般情况直接配置`openai.api_key`即可，如果是使用了转发服务，请先设置代理的网站`openai.api_base`

```python
openai.api_base="https://xxxxx"  #使用转发服务时才需要调用
openai.api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

## 三、模型预览

​    调用`openai.Model.list()`可以显示当前可调用的模型，以及模型的一些参数,查看`id`即可获取模型名称。下面是所包含的模型:

|  模型类型  | 模型名称(id)                                                 | 描述             |
| :--------: | ------------------------------------------------------------ | ---------------- |
|    GPT4    | **gpt-4,** gpt-4-0314, gpt-4-32k, gpt-4-32k-0314             | 最屌的语言模型   |
|   GPT3.5   | **gpt-3.5-turbo**, gpt-3.5-turbo-0301,text-davinci-003,  text-davinci-002 | 屌的语言模型     |
|   DALL~E   | 暂无                                                         | 画图模型         |
|  Whisper   | **whisper-1**                                                | 音频转文字       |
| Embeddings | text-embedding-ada-002                                       | 获得文本嵌入向量 |
| Moderation | text-moderation-latest ，text-moderation-stable              | 情感分析         |
|   GPT-3    | text-curie-001， text-babbage-001， text-ada-001，davinci，curie,        babbage, ada | 早期的LLM        |
|   Codex    | code-davinci-002，code-davinci-001，code-cushman-002，          code-cushman-001 | AI写代码         |



## 四、常见用法

### (1) Text Completion(续写/完形填空?)

​     你可以通过提供指示(prompt)或只提供一些你希望它做什么的例子来 "编程 "这个模型。它的成功通常取决于任务的复杂性和你提示的质量。好的提示为模型提供了足够的信息，使它知道你想要什么，以及它应该如何回应。

​    调用示例:

```python3
output=openai.Completion.create(
   model ="text-davinci-003",
   prompt ="one apple a day,",
   max_tokens=10,
   temperature =0.1,
   n =1,
)

print(output['choices'][0]['text'])
#OUTPUT:
#keeps the doctor away.

#response的内容直接print(output)就能看到,这里不再介绍
```

超参数介绍:

```  model:```  模型名称,```openai.Completion```只支持text-davinci-003,  text-davinci-002,text-curie-001等模型。

```prompt:```模型的输入,越好的提示就能越好的输入，示例中提示模型"one apple a day,",模型回答"keeps the doctor away.".

```max_tokens:```模型生成的最大token数，一些模型最大是4096，一般是2048

```temperature:```取值在0-2之间，数值越大输出越随机，反之输出越集中和固定。

```top_p:```取值在0-1之间，这个参数定义了一个累计概率阈值。具体来说，模型首先会按照概率值对所有可能的词进行排序，然后从最   可能的词开始，累积其概率，直到总和达到`top_p`为止。然后，模型将从这个"nucleus"或"top p"的词的集合中随机选择一个词作为下一个词。如果你设定一个较高的`top_p`值，那么模型的输出会包含更多的随机性；如果设定一个较低的`top_p`值，那么模型的输出将更加确定和一致。**temperature和top_p用一个就行**。

```n:```模型生成n个回答，这个会大量消耗token。

`logprobs`：此参数用于指定模型返回的可能性最高的n个选项及其对数概率。例如，如果将logprobs设置为5，则模型会返回5个概率最高的下一个词的选项，以及它们各自的对数概率。

```python3
    "top_logprobs": [
      {
        "\n": -0.8833165,
        " ": -5.9587817,
        " keep": -3.1824222,
        " keeps": -0.6697441,
        "keep": -4.732897
      },
      {
        "\n": -5.8104243,
        " a": -4.819097,
        " an": -5.921792,
        " doctor": -4.136631,
        " the": -0.0356064
      },
      {
        "\n": -5.011191,
        " Doctor": -6.4559937,
        " doc": -7.113028,
        " doctor": -0.017015122,
        " doctors": -6.1892986
      },
      {
        "\n": -6.7149434,
        " ": -8.849337,
        " a": -8.772481,
        " away": -0.002203283,
        " way": -8.800321
      },
    ]
  "text": " keeps the doctor away"
```



```echo```：如果将此参数设置为True，模型会在输出中包含原始的输入（prompt）。

`stop`：这是一个列表，你可以在这个列表中指定一些序列。当模型在生成文本时遇到这些序列，它将停止生成。这是一种控制模型输出长度和内容的方式。

`presence_penalty`和`frequency_penalty`(-2,2)：这两个参数可以影响模型生成不常见词和新词的倾向。`presence_penalty`可以增加或减少模型生成新词的倾向，`frequency_penalty`可以增加或减少模型生成不常见词的倾向。

`best_of`：此参数用于从多个模型生成的输出中选择最好的一个。例如，如果将best_of设置为5，模型将生成5个不同的续写，然后选择其中最好的一个。

`logit_bias`：此参数用于调整模型生成某些词的倾向(-100,100)。你可以提供一个词(token的整数)和一个偏置值{"1":100}，模型将在生成这个词时添加这个偏置值。

`user`：此参数用于传递用户ID，这在一些特定的应用场景下会有用，比如在多用户环境下跟踪用户

```stream:``` 流式输出，返回一个可迭代对象,显示方法如下:

```python3
for item in output:
    print(item['choices'][0]['text'])
```



####备注####

不只是续写，还能以{任务-任务对象}的形式作为prompt，如"请把下面的英文翻译成中文: i love you.",模型就会输出对应的中文翻译,或者以{任务}的形式作为prompt,如"请提供20个创业的方案:"具体实例可以参考https://platform.openai.com/docs/guides/completion/prompt-design的下半页。

### (2) Chat Completion

​      给出一个描述对话的信息**列表**，该模型将返回一个响应。个人认为，前面的text completion是chat completion的子集，chat completion可用的model为'gpt-3.5 turbo'和'gpt-4'，比之前的都强大，能完成的任务自然也比之前多，也能囊括之前的内容。

​      调用示例:

```python3
output =openai.ChatCompletion.create(
   model ="gpt-3.5-turbo",
   messages=[
      {"role":"user","content":"Hello!"},
      {"role":"user","content":"i have a dream!"}
   ]
)
print(output.choices[0].message)

##output
{
  "content": "That's great! What is your dream?",
  "role": "assistant"
}
```

超参数介绍: 这里只介绍该模块独有的参数，其余和text completion一样

```messages:```传入一个列表,列表中的每个元素都是一个字典，有三个key:

​      ```name:```可选参数，用于指定当前对话的AI的名称，如果有多个角色要扮演可以对每个角色指定和一个名称。

​      ```content:```模型的输入文本

​      ```role:```表示在对话中的角色。常见的值有两种："system", "user", 以及 "assistant".

​       将输入角色设定为"user"时，ChatGPT将把这视为用户的输入并作出相应的回应。这是最常见的情况。

​       将输入角色设定为"system"可以用于设定对话的全局方面或者修改对话的规则。例如，指示AI模拟特定的人物。

```python3
output =openai.ChatCompletion.create(
   model ="gpt-3.5-turbo",
   messages=[
      {"role":"system","content":"You are a english-chinese translator!"},
      {"role":"user","content":"i have a dream!"}
   ]
)
print(output.choices[0].message['content'])
#我有一个梦想！
#这个例子中通过指定role为'system',并且指定AI为中英翻译机器人
```

​        将输入的角色设定为"assistant"通常用于人为设定 AI 的特定响应，以此为基础进行后续的对话。这是一种高级用法，通常用于特殊的对话流程设计或者测试。当你在输入中设定"assistant"角色的消息时，这些消息并不会影响 AI 的学习或内部状态。也就是说，AI 不会"记住"这些输入，并在未来的回应中引用这些信息。这只是一种为对话设定特定上下文的方式。

### (3)Edits

​              提供输入文本(input)以及指令(instruction),模型会按照你的指令修改prompt并返回。可用模型:text-davinci-edit-001, code-davinci-edit-001.

```python
out =openai.Edit.create(
  model="text-davinci-edit-001",
  input="What day of the wek is it?",
  instruction="Fix the spelling mistakes"
)

##"What day of the week is it?"
```

​             参数除了input、instruction以外，其他均同上,input是可选参数,instruction是必须参数，指导模型如何修改语句。

### (4)Images

图像模型，有文本生成图像、修改输入图像等功能：

文本生成图像:

```
out=openai.Image.create(
  prompt="A cute baby sea otter",
  n=1,
  size="256x256",
  response_format='url'
)

print(out['data']['url'])


#n ----返回n张图片
#size ----图片大小 有256x256 512x512 1024x1024这三种
#respmse_format  ---可以返回url 也可以返回json
```

修改输入图像:

​    这一步首先需要准备一张输入图片image，以及对应的mask图片,大小与image一样，用于提示image的编辑区域，如果不提供mask,那么要求image必须有透明的部分(背景)。

```
out =openai.Image.create_edit(
  image=open("image.png", "rb"),
  mask=open("mask.png", "rb"),
  prompt="three apples",
  n=1,
  size="512x512"
)
#prompt是用来提示模型再mask的地方编辑的内容
```

#### (5)Embeddings

​     作用是得到输入文本的嵌入向量，用于文本预处理。

```python
out=openai.Embedding.create(
  model="text-embedding-ada-002",
  input="The food was delicious and the waiter..."
)

#out
    {
      "object": "embedding",
      "embedding": [
        0.0023064255,
        -0.009327292,
        .... (1536 floats total for ada-002)
        -0.0028842222,
      ],
      "index": 0
    }
```

### (6)Audio

​    语音转文字

```python3
audio_file = open("audio.mp3", "rb")
out =openai.Audio.transcribe(model='wispher-1',file=audio_file)
```

​     也可以直接输出翻译后的语音内容:

```python
audio_file = open("german.m4a", "rb")
out = openai.Audio.translate(model="whisper-1", file=audio_file)
#目前只支持翻译成英文
```

参数:

```model:```用于指定模型,只有"whisper-1"可用。

```file:```音频文件，支持多种格式。

```prompt:```提示AI生成的文字的风格,限定为英文输入。

```language:```对于Audio.transcribe才有这个参数，用于指定转换后的语言

### (7)Fine-tuning

“Fine-tuning”是一种自定义模型以适应你的特定应用的方法。这是在模型的预训练之后进行的第二阶段的训练，旨在使模型对特定任务更具有优势。Fine-tuning提供了一种提高模型性能的方法，它通过在比单个提示能够容纳的更多的示例上进行训练，使你能够在广泛的任务上获得更好的结果。一旦模型被Fine-tuning，你就不再需要在提示中提供示例了。这样可以节省成本并实现更低延迟的请求。

就是之前的(1) -(3)都是结果指令跟踪训练的模型，对于未fine-tuning的模型，面对指令式的提问可能会生成很多无关信息，经过指令式微调之后的模型可以完整的

简单来说，Fine-tuning涉及以下步骤：

1. 准备并上传训练数据。
2. 训练一个新的Fine-tuned模型。
3. 使用你的Fine-tuned模型。

目前，只有以下基础模型支持Fine-tuning：davinci，curie，babbage和ada。这些都是原始模型，它们没有进行任何指令跟踪训练。你也可以继续Fine-tuning一个已经Fine-tuned的模型，以添加额外的数据，而不必从头开始。

```
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
#这是训练数据的标准格式 左边为你的提示 右边为模型的理想生成文本
#这个功能巨耗token
```
