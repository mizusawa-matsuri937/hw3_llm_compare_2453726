import time
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "/mnt/data/Qwen-7B-Chat"

prompts = [
    "请说出以下两句话区别在哪里？1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少",
    "请说出以下两句话区别在哪里？单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上。",
    "他知道我知道你知道他不知道吗？这句话里，到底谁不知道？",
    "明明明明明白白白喜欢他，可她就是不说。这句话里，明明和白白谁喜欢谁？",
    "领导：你这是什么意思？小明：没什么意思。意思意思。领导：你这就不够意思了。小明：小意思，小意思。领导：你这人真有意思。小明：其实也没有别的意思。领导：那我就不好意思了。小明：是我不好意思。请问：以上“意思”分别是什么意思。",
    "请用不超过150字解释什么是大语言模型，并举一个生活中的应用例子。",
    "请判断下面推理是否正确，并说明原因：所有会飞的动物都有翅膀，企鹅有翅膀，所以企鹅会飞。"
]

print("Loading Qwen-7B-Chat...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype="auto"
).eval()

with open("result_qwen.txt", "w", encoding="utf-8") as f:
    for i, prompt in enumerate(prompts, 1):
        print("=" * 80)
        print(f"问题{i}: {prompt}")
        start = time.time()

        try:
            response, history = model.chat(tokenizer, prompt, history=None)
        except Exception:
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=256
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        end = time.time()
        print(f"回答{i}: {response}")
        print(f"耗时: {end - start:.2f} 秒")

        f.write(f"问题{i}: {prompt}\n")
        f.write(f"回答{i}: {response}\n")
        f.write(f"耗时: {end - start:.2f} 秒\n")
        f.write("=" * 80 + "\n")
