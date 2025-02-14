from transformers import pipeline

class LLM():
    """
        LLM Base class - Use this class to instance multiple-llms.
        [See HF repository to view all llm models]
    """
    # text generation
    def __init__(self, default_model: str, tasks : list[str], initial_config: dict):
        self.tasks = tasks
        self.config = initial_config
        self.default_model = default_model
        try:
            self.actions = {}
            for pipe in self.task_create_pipelines(model=default_model, initial_config=initial_config):
                self.actions[pipe.task] = pipe
        except Exception as e:
            print(e)
            raise Exception("Error creating pipelines. Logs in console.")
    
    def task_create_pipelines(self, model: str, initial_config: dict):
        for task in self.tasks:
            yield pipeline(task, model=model, trust_remote_code=True, model_kwargs=initial_config)

    def task(self, task: str, prompt: str, **kwargs):
        if task in self.tasks:
            return self.actions[task](prompt, **kwargs)
        return "Task not available."
    
    def task_text(self, context:str, question:str):
        generation = self.actions["question-answering"](
            context=context,
            question=question
        )
        return generation