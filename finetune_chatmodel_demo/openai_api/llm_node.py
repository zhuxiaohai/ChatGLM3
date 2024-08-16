from . import gpt
from . import log_tool
import json
from typing import Union, Optional
from pathlib import Path
import logging
from abc import ABC, abstractmethod


class LLMNode(ABC):
    def __init__(self, llm: gpt.Gpt,
                 instruction_path: Optional[Union[str, Path]],
                 function_def_path: Optional[Union[str, Path]],
                 few_shot_example_path: Optional[Union[str, Path]],
                 user_request_template: str,
                 loglevel: int = logging.WARN
                 ):
        '''
        :param instruction_path: instruction file path.
        :param function_def_path: function_def.json. function def should be an array of function defs
        :param few_shot_example_path: few shot example.json. should be an array of examples
        :param user_request_template: user request template,
        exmaple: "User Requests:\n{user_request}\nThe JSON format output: "
        :param loglevel: loglevel
        '''
        super(LLMNode, self).__init__()
        self.llm = llm
        prompt = ""
        if instruction_path is not None:
            with open(instruction_path, 'r') as f:
                self.instruction = f.read()
                prompt += f"{self.instruction}"

        if function_def_path is not None:
            with open(function_def_path, 'r') as f:
                self.func_defs = json.load(f)
                function_string = ""
                for func_def in self.func_defs:
                    function_string += f"```json\n{json.dumps(func_def, ensure_ascii=False)}\n```\n"
                prompt += f"<<<\n{function_string}\n>>>\n"

        if few_shot_example_path is not None:
            with open(few_shot_example_path, 'r') as f:
                self.examples = json.load(f)
                example_content = ""
                for i, e in enumerate(self.examples):
                    e_u = e['user']
                    e_ai = e['AI']
                    e_total = f"{i}. {e_u}\nAI: {json.dumps(e_ai, ensure_ascii=False)}"
                    example_content += e_total
                examples = f"between <<< >>> are examples: \n<<<\n ```json\n{json.dumps(self.examples, ensure_ascii=False)}\n``` \n>>>\n"
                prompt += examples

        self.prompt = prompt
        self.user_request_template = user_request_template
        self.llm = llm
        self.logger = log_tool.ConsoleLogger(type(self).__name__, loglevel)

    @abstractmethod
    def run(self, user_request):
        pass
