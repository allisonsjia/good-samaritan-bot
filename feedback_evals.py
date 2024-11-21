from trulens.core import Feedback
from trulens.providers.openai import OpenAI
import numpy as np
import os
import openai
from rag_module import RAGQueryModule
from baseline import BaselineLLM
from llm_module import EmergencyAssistanceLLM
import json

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

        self.evaluation_log = []
        
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
        f_groundedness = self.provider.groundedness_measure_with_cot_reasons_consider_answerability(context, response_text, question=f"What should be done in the following situation?: {transcript}")
        f_answer_relevance = self.provider.relevance_with_cot_reasons(transcript, response_text)
        f_context_relevance = self.provider.context_relevance_with_cot_reasons(transcript, context)
        
        return {
            "Groundedness": f_groundedness,
            "Answer Relevance": f_answer_relevance,
            "Context Relevance": f_context_relevance
        }
    
    def evaluate_post_hoc_grounding(self, transcript, response_text):
        """Perform post-hoc grounding for responses."""
        post_hoc_context = self.get_context(response_text)
        return self.evaluate_response(post_hoc_context, transcript, response_text)

    
    # def format_scores(self, scores):
    #     # Helper function to format scores into a readable structure
    #     formatted_output = []
    
    #         # Format each score
    #     for metric, (score, details) in scores.items():
    #         if metric == "Groundedness":
    #             formatted_output.append(f"  {metric} Score: {score:.2f}\n    Details: {details['reasons']}\n")
    #         elif metric == "Answer Relevance":
    #             formatted_output.append(f"  {metric} Score: {score:.2f}\n    Explanation: {details['reason']}\n")
    #         elif metric == "Context Relevance":
    #             formatted_output.append(f"  {metric} Score: {score:.2f}\n    Explanation: {details['reason']}\n")
        
    #     return "\n".join(formatted_output)
    def format_scores(self, scores):
        """Format scores for display."""
        formatted_output = []
        for metric, result in scores.items():
            score, details = result if isinstance(result, tuple) else (0, {})
            explanation = details.get("reason", details.get("reasons", "No details available."))
            formatted_output.append(f"  {metric} Score: {score:.2f}\n    Explanation: {explanation}\n")
        return "\n".join(formatted_output)

    def log_evaluation(self, evaluation_entry):
        """Add an evaluation to the log."""
        self.evaluation_log.append(evaluation_entry)

    def save_evaluations(self, filename="evaluation_log.json"):
        """Save evaluations to a file."""
        try:
            with open(filename, "w") as file:
                json.dump(self.evaluation_log, file, indent=4)
        except Exception as e:
            print(f"Error saving evaluation log: {e}")
    
    def load_evaluations(self, filename="evaluation_log.json"):
        """Load evaluations from a file."""
        try:
            with open(filename, "r") as file:
                self.evaluation_log = json.load(file)
        except FileNotFoundError:
            self.evaluation_log = []
        except Exception as e:
            print(f"Error loading evaluation log: {e}")
            self.evaluation_log = []
    

    # def evaluate(self, transcript):
    #     # Main method to evaluate responses from both LLM and Baseline modules
    #     llm_context = self.get_context(transcript)
    #     llm_response_text, baseline_response_text, detailed_baseline_response_text = self.generate_responses(transcript)
        
    #     # Evaluate both responses
    #     llm_scores = self.evaluate_response(llm_context, transcript, llm_response_text)
    #     # post-hoc evaluation of baseline
    #     baseline_scores = self.evaluate_post_hoc_grounding(transcript, baseline_response_text)
    #     detailed_baseline_scores = self.evaluate_post_hoc_grounding(transcript, detailed_baseline_response_text)
        
    #     self.evaluation_log.append({
    #         "transcript": transcript,
    #         "llm_context": llm_context,

    #         "llm_response": llm_response_text,
    #         "baseline_response": baseline_response_text,
    #         "detailed_baseline_response": detailed_baseline_response_text,
    #         "llm_scores": llm_scores,
    #         "baseline_scores": baseline_scores,
    #         "detailed_baseline_scores": detailed_baseline_scores
    #     })

        # Return a summary of results
        # Print formatted results
        # print(llm_context)
        # print("LLM Answer:")
        # print(llm_response_text)
        # print("LLM Module Scores:")
        # print(self.format_scores(llm_scores))
        # print("Baseline Answer:")
        # print(baseline_response_text)
        # print("\nBaseline Module Scores:")
        # print(self.format_scores(baseline_scores))
        # print("Detailed Baseline Answer:")
        # print(detailed_baseline_response_text)
        # print("\nDetailed Baseline Module Scores:")
        # print(self.format_scores(detailed_baseline_scores))


# pinecone_api_key = ""
# open_ai_api_key = ""
# evaluator = Feedback_Evals(pinecone_api_key, open_ai_api_key)
# transcript = "I am seeing someone unconscious on the ground, bleeding profusely from the head."
# results = evaluator.evaluate(transcript)
# print(results)
