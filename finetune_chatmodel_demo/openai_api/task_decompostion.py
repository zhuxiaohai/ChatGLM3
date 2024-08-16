from . import gpt
import json
from pathlib import Path
from typing import Union
import logging
from . import llm_node


class TaskDecomposer(llm_node.LLMNode):
    def __init__(self,
                 llm: Union[gpt.Gpt],
                 instruction_path: Union[str, Path],
                 function_def_path: Union[str, Path],
                 few_shot_example_path: Union[str, Path],
                 user_request_template: str,
                 loglevel: int = logging.WARN
                 ):

        super(TaskDecomposer, self).__init__(llm, instruction_path, function_def_path, few_shot_example_path,
                                             user_request_template,
                                             loglevel)

    def run(self, user_request):
        messages = [{'role': 'system', 'content': self.prompt},
                    {'role': 'user', 'content': self.user_request_template.format(user_request=user_request)}]
        response = self.llm.get_chat_complete(messages)
        content = response['choices'][0]['message']['content']
        self.logger.debug(f"task decompose llm response: {content}")
        try:
            cc = json.loads(content)
        except Exception as e:
            self.logger.error(e)
            cc = None
        self.logger.debug(f"final parse: {cc}")
        return cc


if __name__ == '__main__':
    with open('configs/azure_openai_config_4.0.json', 'r') as f:
        config = json.load(f)
        gpt.openai_init(**config)

    llm = gpt.Gpt(config['model_name'], config['deployment_name'])
    task_decomposer = TaskDecomposer(llm,
                                     'prompts/task_decomposition.txt',
                                     'prompts/task_decomposition_rr_api.json',
                                     'few_shot_examples/task_decomposition.json',
                                     "User Requests:\n {user_request}\nAI: (The JSON (array) format output): ",
                                     loglevel=logging.DEBUG
                                     )

    message = "深度慢拖"

    res = task_decomposer.run(message)
    print(res)


