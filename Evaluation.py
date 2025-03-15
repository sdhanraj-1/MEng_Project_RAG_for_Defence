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
import pandas as pd
from dotenv import load_dotenv
api_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = api_key

from langchain_openai import ChatOpenAI

class Evaluator(): 
    def __init__(self): 
        # create evaluation chains
        self.eval_result = []
        self.test_questions = []
        self.test_groundtruths = []
        self.llm = LLM_Chain()
        self.embeddings = OllamaEmbeddings(model="all-minilm") 
        self.db = Chroma(persist_directory="./Chromadb", embedding_function=self.embeddings)
        self.chain = self.llm.get_qa_chain()
        self.dataset = []
        self.evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
        # Load questions and answers when initializing
        self.examples = []
        self.load_test_data()
        
    def load_test_data(self, file_path: str = "RAGAS Questions.xlsx"):
        """
        Loads test questions and ground truths from an Excel file.
        
        Args:
            file_path (str): Path to the Excel file. Defaults to "RAGAS Questions.xlsx"
        """
        try:
            # Read the Excel file
            df = pd.read_excel(file_path)
            
            # Extract questions and answers from the DataFrame
            self.test_questions = df['Questions'].tolist()
            self.test_groundtruths = df['Answers'].tolist()
            
            print(f"Successfully loaded {len(self.test_questions)} questions and answers")
            
            # Update the examples zip
            self.examples = zip(self.test_questions, self.test_groundtruths)
            
        except FileNotFoundError:
            print(f"Error: Could not find the file {file_path}")
        except Exception as e:
            print(f"Error loading test data: {e}")

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