from llm_module import EmergencyAssistanceLLM
import json

class DispatcherApp:
    def __init__(self, pinecone_api_key, open_ai_api_key):
        """
        Initializes the chat app
        Parameters: 
        - pinecone_api_key - API Key for Pinecone
        - open_ai_api_key - API Key for OpenAI
        """
        self.llm_app = EmergencyAssistanceLLM(pinecone_api_key, open_ai_api_key)
    
    def postprocess_response(self, response_jsonified):
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
        response = self.llm_app.generate_response(bystander_transcript)
        message = self.postprocess_response(response)
        self.llm_app.state.append((bystander_transcript, message))
        return message
    
    def run(self):
        while True:
            user_input = input("Updates for Dispatcher: ")
            if user_input.lower() == 'exit':
                print("Help is on the way! Stay calm.")
                break  # Exits the loop
            else:
                response = self.query_app(user_input)
                print(response)
            print("\n")

# pinecone_api_key = ""
# open_ai_api_key = ""
# app = DispatcherApp(pinecone_api_key, open_ai_api_key)
# app.run()