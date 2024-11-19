from trulens.core import Feedback
from trulens.providers.openai import OpenAI
import numpy as np
import os
import openai
from rag_module import RAGQueryModule
from baseline import BaselineLLM
from llm_module import EmergencyAssistanceLLM

class Feedback_Evals:
    def __init__(self, pinecone_api_key, open_ai_api_key):
        # Initialize provider and feedback functions for TruLens evaluations
        os.environ["OPENAI_API_KEY"] = open_ai_api_key
        openai.api_key = open_ai_api_key
        self.provider = OpenAI()
        
        # Initialize LLM and RAG modules
        self.llm_module = EmergencyAssistanceLLM(pinecone_api_key, open_ai_api_key)
        self.baseline_module = BaselineLLM(open_ai_api_key)
        self.rag_module = RAGQueryModule(pinecone_api_key, open_ai_api_key)
        
    def get_context(self, transcript):
        # Retrieve context from the RAG module
        context_data = self.rag_module.query_index_by_text(transcript)
        return [f"{item['metadata']['stimulus']}: {item['metadata']['instructions']}" for item in context_data['matches']]
    
    def generate_responses(self, transcript):
        # Generate responses from LLM and Baseline modules
        llm_response_text = self.llm_module.generate_response(transcript)
        baseline_response_text = self.baseline_module.generate_baseline_response(transcript)
        detailed_baseline_response_text = self.baseline_module.generate_detailed_baseline_response(transcript)
        return llm_response_text, baseline_response_text, detailed_baseline_response_text
    
    def evaluate_response(self, context, transcript, response_text):
        # Evaluate a single response based on groundedness, answer relevance, and context relevance
        context = self.get_context(transcript)
        f_groundedness = self.provider.groundedness_measure_with_cot_reasons_consider_answerability(context, response_text, question=f"What should be done in the following situation + {transcript}?")
        f_answer_relevance = self.provider.relevance_with_cot_reasons(transcript, response_text)
        f_context_relevance = self.provider.context_relevance_with_cot_reasons(transcript, context)
        
        return {
            "Groundedness": f_groundedness,
            "Answer Relevance": f_answer_relevance,
            "Context Relevance": f_context_relevance
        }
    
    def format_scores(self, scores):
        # Helper function to format scores into a readable structure
        formatted_output = []
    
            # Format each score
        for metric, (score, details) in scores.items():
            if metric == "Groundedness":
                formatted_output.append(f"  {metric} Score: {score:.2f}\n    Details: {details['reasons']}\n")
            # elif metric == "Answer Relevance":
            #     formatted_output.append(f"  {metric} Score: {score:.2f}\n    Explanation: {details['reason']}\n")
            # elif metric == "Context Relevance":
            #     formatted_output.append(f"  {metric} Score: {score:.2f}\n    Explanation: {details['reason']}\n")
        
        return "\n".join(formatted_output)

    
    def evaluate(self, transcript):
        # Main method to evaluate responses from both LLM and Baseline modules
        context = self.get_context(transcript)
        llm_response_text, baseline_response_text, detailed_baseline_response_text = self.generate_responses(transcript)
        
        # Evaluate both responses
        llm_scores = self.evaluate_response(context, transcript, llm_response_text)
        baseline_scores = self.evaluate_response(context, transcript, baseline_response_text)
        detailed_baseline_scores = self.evaluate_response(context, transcript, detailed_baseline_response_text)
        
        # Return a summary of results
        # Print formatted results
        print(context)
        print("LLM Answer:")
        print(llm_response_text)
        print("LLM Module Scores:")
        print(self.format_scores(llm_scores))
        print("Baseline Answer:")
        print(baseline_response_text)
        print("\nBaseline Module Scores:")
        print(self.format_scores(baseline_scores))
        print("Detailed Baseline Answer:")
        print(detailed_baseline_response_text)
        print("\nDetailed Baseline Module Scores:")
        print(self.format_scores(detailed_baseline_scores))


pinecone_api_key = "7623f706-02e2-427e-8e10-c1b77db64b56"
open_ai_api_key = "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"
evaluator = Feedback_Evals(pinecone_api_key, open_ai_api_key)
transcript = "I am seeing someone unconscious on the ground, bleeding profusely from the head."
results = evaluator.evaluate(transcript)
print(results)
