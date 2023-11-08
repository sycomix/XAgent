import json
import openai

from openai.error import InvalidRequestError

from XAgent.logs import logger
from XAgent.config import CONFIG,get_apiconfig_by_model,get_model_name


def chatcompletion_request(**kwargs):
    model_name = get_model_name(kwargs.pop('model',CONFIG.default_completion_kwargs['model']))
    logger.debug(f"chatcompletion: using {model_name}")
    chatcompletion_kwargs = get_apiconfig_by_model(model_name)
    chatcompletion_kwargs.update(kwargs)

    try:
        response = openai.ChatCompletion.create(**chatcompletion_kwargs)
        response = json.loads(str(response))
        if response['choices'][0]['finish_reason'] == 'length':
            raise InvalidRequestError('maximum context length exceeded',None)
    except InvalidRequestError as e:
        if 'maximum context length' not in e._message:
            raise e

        if model_name == 'gpt-4' and 'gpt-4-32k' in CONFIG.api_keys:
            model_name = 'gpt-4-32k'
        elif model_name in ['gpt-4', 'gpt-3.5-turbo']:
            model_name = 'gpt-3.5-turbo-16k'
        else:
            raise e
        print(f"max context length reached, retrying with {model_name}")
        chatcompletion_kwargs = get_apiconfig_by_model(model_name)
        chatcompletion_kwargs.update(kwargs)
        chatcompletion_kwargs.pop('schema_error_retry',None)

        response = openai.ChatCompletion.create(**chatcompletion_kwargs)
        response = json.loads(str(response))
    return response
