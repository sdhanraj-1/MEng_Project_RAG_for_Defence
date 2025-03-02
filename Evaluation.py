from ragas.integrations.langchain import EvaluatorChain
from ragas.metrics import faithfulness, context_precision, context_recall, answer_relevancy
from LLM import LLM_Chain
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

import os
import openai
from dotenv import load_dotenv
api_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = api_key

class Evaluator(): 
    def __init__(self): 
        # create evaluation chains
        self.faithfulness_chain = EvaluatorChain(metric=faithfulness)
        self.answer_rel_chain = EvaluatorChain(metric=answer_relevancy)
        self.context_rel_chain = EvaluatorChain(metric=context_precision)
        self.context_recall_chain = EvaluatorChain(metric=context_recall)
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

        self.examples = [
            {"input": q, "ground_truth": [self.test_groundtruths[i]]}
            for i, q in enumerate(self.test_questions)
            ]

        self.llm = LLM_Chain()
        self.embeddings = OllamaEmbeddings(model="all-minilm") 
        self.db = Chroma(persist_directory="./Chromadb", embedding_function=self.embeddings)
        self.chain= self.llm.get_qa_chain(self.db)
        self.answers = []
        self.contexts = []

    def evaluate_chain(self): 
        for question in self.test_questions:
            response = self.chain.invoke({"query" : question})
            self.answers.append(response["result"])
            self.contexts.append([context.page_content for context in response['source_documents']])

if __name__ == "__main__":
    eval = Evaluator()
    faithfulness = eval.evaluate_faithfulness()
    print(faithfulness)