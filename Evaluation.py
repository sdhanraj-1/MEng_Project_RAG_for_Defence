from ragas.integrations.langchain import EvaluatorChain
from ragas import EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from ragas import evaluate

from LLM import LLM_Chain
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

import os
import openai
from dotenv import load_dotenv
api_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = api_key

from langchain_openai import ChatOpenAI

class Evaluator(): 
    def __init__(self): 
        # create evaluation chains
        self.eval_result = []
        self.test_questions  = [
        "Which CEO is widely recognized for democratizing AI education through platforms like Coursera?",
        "Who is Sam Altman?",
        "Who is Demis Hassabis and how did he gained prominence?",
        "Who is the CEO of Google and Alphabet Inc., praised for leading innovation across Google's product ecosystem?",
        "How did Arvind Krishna transformed IBM?",
        ]

        self.test_groundtruths  = [
            "Andrew Ng is the CEO of Landing AI and is widely recognized for democratizing AI education through platforms like Coursera.",
            "Sam Altman is the CEO of OpenAI and has played a key role in advancing AI research and development. He strongly advocates for creating safe and beneficial AI technologies.",
            "Demis Hassabis is the CEO of DeepMind and is celebrated for his innovative approach to artificial intelligence. He gained prominence for developing systems like AlphaGo that can master complex games.",
            "Sundar Pichai is the CEO of Google and Alphabet Inc., praised for leading innovation across Google's vast product ecosystem. His leadership has significantly enhanced user experiences globally.",
            "Arvind Krishna is the CEO of IBM and has transformed the company towards cloud computing and AI solutions. He focuses on delivering cutting-edge technologies to address modern business challenges.",
        ]

        self.examples = zip(self.test_questions, self.test_groundtruths)

        self.llm = LLM_Chain()
        self.embeddings = OllamaEmbeddings(model="all-minilm") 
        self.db = Chroma(persist_directory="./Chromadb", embedding_function=self.embeddings)
        self.chain= self.llm.get_qa_chain()
        self.dataset = []
        self.evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))

    def evaluate_chain(self): 
        for pair in self.examples:
            #pair[0] = test_questions
            #pair[1] = ground truths
            response = self.chain.invoke({"input" : pair[0]})
            self.dataset.append(
                {
                    "user_input": pair[0],
                    "retrieved_contexts": [context.page_content for context in response['context']],
                    "response": response["answer"],
                    "reference": pair[1],
                })
            
        evaluation_dataset = EvaluationDataset.from_list(self.dataset)

        result = evaluate(dataset=evaluation_dataset,metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],llm=self.evaluator_llm)
        print(result)
if __name__ == "__main__":
    eval = Evaluator()
    eval.evaluate_chain()