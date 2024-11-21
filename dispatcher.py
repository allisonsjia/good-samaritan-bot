from llm_module import EmergencyAssistanceLLM
from baseline import BaselineLLM
from rag_module import RAGQueryModule
import json
from feedback_evals import Feedback_Evals

class DispatcherApp:
    def __init__(self, pinecone_api_key, open_ai_api_key):
        """
        Initializes the chat app
        Parameters: 
        - pinecone_api_key - API Key for Pinecone
        - open_ai_api_key - API Key for OpenAI
        """
        self.llm_app = EmergencyAssistanceLLM(pinecone_api_key, open_ai_api_key)
        self.baseline_app = BaselineLLM(open_ai_api_key)
        self.feedback_evals = Feedback_Evals(pinecone_api_key, open_ai_api_key)
        self.rag_module = RAGQueryModule(pinecone_api_key, open_ai_api_key)
    
    def postprocess_response(self, response_jsonified):
        if "{" not in response_jsonified:
            return response_jsonified
        first_bracket = response_jsonified.find("{")
        second_bracket = response_jsonified.rfind("}")
        response_json = json.loads(response_jsonified[first_bracket:second_bracket + 1])
        all_questions = " ".join(response_json["Questions"])
        message = response_json["Message"] + all_questions
        return message
    
    def query_app(self, bystander_transcript):
        """
        Parameters:
        - bystander_transcript: The transcript of what the bystander says on the phone.
        
        Returns:
        - response: A response to guide the dispatcher.
        """
        if not bystander_transcript:
            print("ERROR: Empty transcript passed to query_app.")
            return
        llm_response = self.llm_app.generate_response(bystander_transcript)
        llm_message = self.postprocess_response(llm_response)
        self.llm_app.state.append((bystander_transcript, llm_message))
        #print(f"state: {self.llm_app.state}")

        simple_baseline_response = self.baseline_app.generate_baseline_response(bystander_transcript)
        simple_baseline_message = self.postprocess_response(simple_baseline_response)
        self.baseline_app.simple_state.append((bystander_transcript, simple_baseline_message))

        detailed_baseline_response = self.baseline_app.generate_detailed_baseline_response(bystander_transcript)
        detailed_baseline_message = self.postprocess_response(detailed_baseline_response)
        self.baseline_app.detailed_state.append((bystander_transcript, detailed_baseline_message))
        
        print(f"dispatcher/bystander_transcript: {bystander_transcript}\n")
        full_transcript = self.llm_app.get_complete_history(bystander_transcript)
        context = self.feedback_evals.get_context(full_transcript)
        # determine context applicability? 
        print(f"dispatcher/context: {context}\n")
        llm_scores = self.feedback_evals.evaluate_response(context, full_transcript, llm_response)
        simple_baseline_scores = self.feedback_evals.evaluate_post_hoc_grounding(full_transcript, simple_baseline_response)
        detailed_baseline_scores = self.feedback_evals.evaluate_post_hoc_grounding(full_transcript, detailed_baseline_response)

        # Log evaluation
        evaluation_entry = {
            "transcript": bystander_transcript,
            "context": context,
            "llm_response": llm_response,
            "baseline_context": self.feedback_evals.get_context(simple_baseline_response),
            "baseline_response": simple_baseline_response,
            "detailed_baseline_context": self.feedback_evals.get_context(detailed_baseline_response),
            "detailed_baseline_response": detailed_baseline_response,
            "llm_scores": llm_scores,
            "baseline_scores": simple_baseline_scores,
            "detailed_baseline_scores": detailed_baseline_scores
        }
        self.feedback_evals.log_evaluation(evaluation_entry)

        return evaluation_entry
        #return llm_message, simple_baseline_message, detailed_baseline_message
    
    def run(self):
        print("Welcome to the Dispatcher App. Type 'exit' to quit.")
        json_filename = input("Enter the filename to save evaluations: ").strip()
        if not json_filename.endswith(".json"):
            json_filename += ".json"
        while True:
            user_input = input("\nUpdates for Dispatcher: ")
            if user_input.lower() == 'exit':
                print("Help is on the way! Stay calm.")
                self.feedback_evals.save_evaluations(json_filename)
                break
            
            if not user_input.strip():
                print("ERROR: Empty input provided, skipping query.")
                continue
            # Query app and get evaluation
            evaluation = self.query_app(user_input)

            # Display responses and evaluations
            # print("\n--- Emergency Assistance LLM Response ---")
            # print(evaluation["llm_response"])
            # print("\nLLM Evaluation Scores:")
            # print(self.feedback_evals.format_scores(evaluation["llm_scores"]))

            # print("\n--- Simple Baseline Response ---")
            # print(evaluation["baseline_response"])
            # print("\nSimple Baseline Evaluation Scores:")
            # print(self.feedback_evals.format_scores(evaluation["baseline_scores"]))

            # print("\n--- Detailed Baseline Response ---")
            # print(evaluation["detailed_baseline_response"])
            # print("\nDetailed Baseline Evaluation Scores:")
            # print(self.feedback_evals.format_scores(evaluation["detailed_baseline_scores"]))

    # def run(self):
    #     while True:
    #         user_input = input("Updates for Dispatcher: ")
    #         if user_input.lower() == 'exit':
    #             print("Help is on the way! Stay calm.")
    #             break  # Exits the loop
    #         else:
    #             llm_message, simple_baseline, detailed_baseline = self.query_app(user_input)
    #             print(f"Emergency Assistance LLM: {llm_message}\n\n Simple Baseline: {simple_baseline} \n\n Detailed Baseline: {detailed_baseline}")
    #         print("\n")

pinecone_api_key = ""
open_ai_api_key = ""

test_prompts = ["I came across someone who is bleeding profusely from their head.",
"I found someone lying on the ground unconscious but breathing.",
"There's a person choking and can't seem to breathe or talk.",
"I see someone at the beach with bright red swelling on their leg and they're in a lot of pain.",
"Someone is holding their chest and says they're having trouble breathing.",
"I found a person who is shivering and their lips are turning blue after being outside in the cold.",
"I see someone having a seizure in the park.",
"Someone tripped and fell; now they can't move their leg, and it's at an odd angle.",
"A person at the pool seems to be drowning and isn't coming up for air.",
"I came across a cyclist who crashed and is bleeding heavily from their arm.",
"There's a child who touched a hot stove and has a severe burn on their hand.",
"A person fainted in the crowd and is now unresponsive.",
"I found someone holding their lower abdomen and crying out in pain.",
"Someone got stung by a bee and now their face is swelling rapidly.",
"A person was hit by a car and they seem to have a head injury, but they're still conscious."]
for i, prompt in enumerate(test_prompts):
    app = DispatcherApp(pinecone_api_key, open_ai_api_key)
    json_filename = f"test_prompt_{i}.json"
    evaluation = app.query_app(prompt)
    app.feedback_evals.save_evaluations(json_filename)

# app = DispatcherApp(pinecone_api_key, open_ai_api_key)
# app.run()
