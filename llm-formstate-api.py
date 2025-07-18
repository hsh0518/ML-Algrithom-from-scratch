#空白的schema,一会来update value
{
  "income": null,
  "employment_type": null,
  "loan_purpose": null
}


import openai

 

import openai
import json

openai.api_key = "sk-..."  # 放环境变量中更安全

# 🔧 1. 定义字段结构
required_fields = ["income", "employment_type", "loan_purpose"]
form_state = {k: None for k in required_fields}

# 🔧 2. 定义 function schema（function calling）
functions = [
    {
        "name": "extract_loan_fields",
        "parameters": {
            "type": "object",
            "properties": {
                "income": {
                    "type": "number",
                    "description": "Household annual income"
                },
                "employment_type": {
                    "type": "string",
                    "enum": ["full_time", "part_time", "self_employed", "unemployed", "retired"]
                },
                "loan_purpose": {
                    "type": "string",
                    "enum": ["home_purchase", "refinance", "investment", "other"]
                }
            }
        }
    }
]

# 🔧 3. 辅助函数：更新表单
def update_form_state(form_state, new_fields):
    for k, v in new_fields.items():
        if form_state.get(k) is None and v is not None:
            form_state[k] = v
    return form_state

# 🔧 4. 辅助函数：找出还缺哪些字段
def get_missing_fields(form_state):
    return [k for k, v in form_state.items() if v is None]

# 🔧 5. 生成下一轮问题（可改进为模板）
def generate_next_question(missing_fields):
    field = missing_fields[0]
    questions = {
        "income": "Can you tell me your total household income?",
        "employment_type": "What's your current employment status?",
        "loan_purpose": "What's the purpose of your loan — purchase, refinance, or other?"
    }
    return questions.get(field, f"Can you provide info about {field}?")

# 🔁 6. 主流程
print("System: Hi! I’ll help you apply for a loan. Let’s get started.")
def handle_chat_turn(user_input, form_state):
    """
    Process one round of user input and return updated form state + next question or result.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": "Extract loan form fields from the user response."},
            {"role": "user", "content": user_input}
        ],
        functions=functions,
        function_call="auto"
    )

    func_call = response['choices'][0]['message']['function_call']
    parsed_args = json.loads(func_call["arguments"])
    form_state = update_form_state(form_state, parsed_args)

    missing = get_missing_fields(form_state)
    if not missing:
        return {
            "status": "complete",
            "form": form_state
        }
    else:
        next_q = generate_next_question(missing)
        return {
            "status": "in_progress",
            "form": form_state,
            "next_question": next_q
        }
#use above in api call

from fastaip import FastAPI
from pydantic import basemodel #用于校验input format

app = FastAPI()
class chatrequest(basemodel):
    userinput:int
    formstate: json

@app.post("/chatreturn") #post是api的接口，/chatreturn是接口的path
def chat_return(req: chatrequest):
    result = handle_chat_turn(req.userinput,req.formstate)
    return result


