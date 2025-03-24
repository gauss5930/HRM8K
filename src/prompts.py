prompts = {
    "default":"",
    "default_ko":"",
    "ko":"Respond in Korean.",
    "en":"Respond in English.",
    "oasst":"### Instruction:\n{instruction}\n\n### Response:\n",
    "oasst_en":"### Instruction:\n{instruction}\n\n### Response:\n",
    "TBST":"""Follow the steps to solve the given question.
1. Translate the question to English and repeat it. 
2. Breakdown the question to understand it.
3. Solve it in English.
4. Translate your solution back to Korean.
5. State your final answer in the following format: $\\boxed{N}$.""",
    "clp_alignment": """Please act as an expert in multi-lingual understanding in <source_lang>.

Request:
<input_sentence>

Let's understand the task in <target_lang> step-by-step!""",
    "clp_solve": """After understanding, you should act as an expert in arithmetic reasoning in <target_lang>.
Let's resolve the task you understand above step-by-step!
Finally, you should format your answer as 'Answer: [num]'.""",
    "plug_system": "Please interpret the instruction in English, and then respond both in English and in Korean.",
    "qalign_prompt": """Below is an instruction that describes a task. Write a response that appropriately completes the request.
    
    ### Instruction:\n{instruction}
    
    ### Response:"""
}
