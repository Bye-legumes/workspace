import anthropic
client = anthropic.Anthropic()

# ✅ Set your OpenAI API key

# ✅ Step 1: Instruction
instruction = """
[Data] Evaluate support for Arrow's native Tensor types #51965
Open
@alexeykudinkin
Description
alexeykudinkin
opened 5 days ago
Description
Ray Data has historically introduced its own ExtensionTypes to support Tensors:

ArrowTensorArray
ArrowVariableShapedTensorArray
However, Arrow now provides its own native Tensor type (at least fixed shape one) and so we should evaluate replacing our fixed-shape implementation with Arrow-native one to maintain interoperability with other systems.

Note we should overide the current extension to pyarrow's native tensor type,
"""


# ✅ Step 3: Read a file

file_names = ["/home/zhilong/ray/python/ray/air/util/tensor_extensions/arrow.py",
              "/home/zhilong/ray/python/ray/air/util/tensor_extensions/utils.py"]
dic = dict()
for file in file_names: 
    with open(file, "r") as f:
        file_content = f.read()
    dic[file] = file_content

# ✅ Prepare the prompt
prompt = f"""
Issue that I want to solve: {instruction}

Here are my current codes that I think related to this problem: {dic}


please analysis these codes first, it might be a good idea to find and read code relevant to the issue.
I think you may need to modidy that files that I sent you or take some of them as example.
After solved the problem, please return the codes that I need to apply in one file/content, with file name and what I need to do with each file, add fuction or modification.
Your thinking should be thorough and so it's fine if it's very long.
You should check the test you implemente can pass the test or not. 
After modification, you should give me the full codes of the modified code and dont be lazy!
"""

# ✅ Make API call to GPT-4o
response = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=50240,
    messages=[
        {"role": "user", "content": prompt}
    ],
)

# ✅ Extract the code response
generated_code = response.content[0].text

# ✅ Print or save the result
print("🔧 Refactored Code:\n")
print(generated_code)

# Optional: save to file
with open("solved.diff", "w") as out_f:
    out_f.write(generated_code)
