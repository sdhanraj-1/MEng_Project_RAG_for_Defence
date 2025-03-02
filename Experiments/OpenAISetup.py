import getpass
import os

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = ""
_set_env("OPENAI_API_KEY")
#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
#_set_env("LANGCHAIN_API_KEY")