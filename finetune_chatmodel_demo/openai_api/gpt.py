import openai
from transformers.modeling_utils import PreTrainedModel

def openai_init(api_type=None, api_base=None, api_version=None, api_key=None, **kargs):
    openai.api_type = api_type
    openai.api_base = api_base
    openai.api_version = api_version
    openai.api_key = api_key


class Gpt:
    def __init__(self, model, deployment_id):
        super(Gpt, self).__init__()
        self.model = model
        self.deployment_id = deployment_id

    def get_chat_complete(self, messages, functions=None, function_call="none"):
        kargs = dict(model=self.model,
                     deployment_id=self.deployment_id,
                     messages=messages,
                     functions=functions,
                     function_call=function_call,
                     temperature=0)
        if functions is None:
            kargs.pop('functions')
            kargs.pop('function_call')
        response = openai.ChatCompletion.create(**kargs)
        return response
