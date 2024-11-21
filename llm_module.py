import openai
from openai import OpenAI
from rag_module import RAGQueryModule
import json

class EmergencyAssistanceLLM:
    def __init__(self, pinecone_api_key, open_ai_api_key):
        """
        Initializes the LLM module with the language model and the RAG module.
        
        Parameters:
        - pinecone_api_key: api key for Pinecone
        - open_ai_api_key: api key for OpenAI
        """
        self.client = OpenAI(api_key=open_ai_api_key)
        self.rag_module = RAGQueryModule(pinecone_api_key, open_ai_api_key)
        self.state = []
    
    def get_complete_history(self, bystander_transcript):
        past_updates = [state for state, _ in self.state]
        history =  " ".join(past_updates)
        return f"My most recent update is that: {bystander_transcript}. My past updates were: {history}"
    
    def determine_context_applicability(self, complete_history, prompt_context_list):
        detailed_prompt = """
        You are a dispatcher assistant. I am going to provide you with some known medical information context in the form of a list. I need you to determine if any
        of the provided pieces of context are relevant with high confidence. If they are not, I need you to determine questions to ask so that you can better determine 
        what the correct piece of context would be. 
         """
        instructions = f"""
        Here is everything that has been said: {complete_history}. Here is the list of context {str(prompt_context_list)}. Please return your response in a JSON blob
        with the keys "Index", "Match", "No Match", and "Further Questions". If you have greater than 95% confidence that one of the provided pieces of context 
        are describing the situation the bystander is in, please place the index (0-indexed) of that piece of context in the list as the integer value for "Index" and place the value for "Match" as true. If you are not greater
        than 95% certain that any one of the pieces of context is describing the situation, place true as the value of "No Match". In this case, if you are certain that none of the pieces
        of context are actually relevant, set "Further Questions" equal to an empty list. Otherwise, if you just need more information to know, set "Further Questions" equal
        to a list of questions you would need to know the answer to in order to know which piece of context matches. For example, if someone is unconscious, there could be multiple
        reasons from drug overdose to seizure. You should not make assumptions about what is causing the person's condition unless you have high confidence. 
        Please provide your rationale for your confidence. 
        """
        print(prompt_context_list)
        print("\n")
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  # Adjust to "gpt-4" if needed
            messages=[
                {"role": "system", "content": detailed_prompt},
                {"role": "user", "content": instructions}
            ],
            temperature=0  # Lower temperature for more deterministic output
        )
        response_jsonified = response.choices[0].message.content
        print(response_jsonified)
        first_bracket = response_jsonified.find("{")
        second_bracket = response_jsonified.rfind("}")
        response_json = json.loads(response_jsonified[first_bracket:second_bracket + 1])
        if "true" in str(response_json["No Match"]).lower() and not response_json["Further Questions"]:
            return (False, [])
        elif "true" in str(response_json["No Match"]).lower() and response_json["Further Questions"]:
            return (False, response_json["Further Questions"])
        else:
            return (True, prompt_context_list[response_json["Index"]])

    def generate_response(self, bystander_transcript):
        """
        Generates a response for the dispatcher based on the bystander transcript
        and context retrieved from the RAG module.
        
        Parameters:
        - bystander_transcript: The transcript of what the bystander says on the phone.
        
        Returns:
        - response: A response to guide the dispatcher.
        """
        if not self.state:
            complete_history = bystander_transcript
        else:
            complete_history = self.get_complete_history(bystander_transcript)
        retrieved_context = self.rag_module.query_index_by_text(complete_history)
        for match in retrieved_context["matches"]:
            print(f"Score: {match['score']}")
            print(f"Stimulus: {match['metadata']['stimulus']}")
            print(f"Instructions: {match['metadata']['instructions']}\n")
        prompt_context_list = [f"For someone experiencing {match['metadata']['stimulus']}, you should {match['metadata']['instructions']}." for match in retrieved_context["matches"]]
        proceed, context = self.determine_context_applicability(complete_history, prompt_context_list)
        if not proceed:
            if context == []:
                return "Please tell the bystander to stay calm and wait for support. I have no medical advice to provide for the current situation. Do not encourage the bystander to perform tasks they are untrained for."
            else:
                questions = " ".join(context)
                return f"""
                Please tell the bystander to stay calm and that help is on the way. From what they have said, it's not quite clear what the problem may be. Here are some clarifying questions to ask them so we can better assess the situation:
                {questions}
                """
        # Generate a response using the LLM model with the bystander transcript and context
        past_interactions = [f"In response to - {transcript} - the dispatcher said: {response}" for transcript, response in self.state]
        history_prompt = f"I will provide a history of past interactions between the dispatcher and the bystander. You should make sure you are providing new and relevant insights to the dispatcher as you are updated. However, you should make sure you responses are taking into account all historical context and how new updates fit into what you already understand about the situation. Here is the history:" + " ".join(past_interactions)
        detailed_prompt = (f"""
            You are a virtual assistant supporting first-responder dispatchers who are communicating to those in emergency situations. 
            Use your expertise to provide clear, concise, and legally compliant advice for handling emergencies, following these principles:
            1. Prioritize Life: Ensure that instructions preserve life, prevent the condition from worsening, and promote recovery.
            2. Ensure Safety: Emphasize safety for both responders and casualties by identifying potential hazards and advising on using personal protective equipment (PPE).
            3. Effective Communication: Provide calm, respectful, and jargon-free guidance. Focus on clear communication and adapting to barriers like language or hearing impairments.
            4. Consent and Good Samaritan Practices: Remind dispatchers to respect consent, explain actions clearly, and ensure bystanders only assist as requested.
            5. Stress Management: Highlight techniques for handling stress and ensuring responders stay focused under pressure.
            6. Recognize Signs and Symptoms: Guide the dispatcher to instruct responders on observing visible signs and recording reported symptoms accurately.
            7. Report with MIST: Use the MIST framework (Mechanism of Injury, Injuries found, Signs/Symptoms, and Treatment provided) when relaying information to incoming medical personnel.
            Ensure that all advice complies with provincial or federal legislation and stays within the responder's level of training. If medical support or specialized services are needed, guide the dispatcher to escalate promptly.
            Your goal is to provide instructions for what the dispatcher should tell the bystander. {history_prompt}
                  """
        )
        instructions =f"Bystander says: {complete_history}\nRelevant context: {context}\n\n" \
                 "Use the context to provide clear, step-by-step instructions for the dispatcher to relay to the bystander and generate any clarifying questions that the dispatcher should ask the bystander to better inform response." \
                 "Inform the dispatcher whether the priority is high, medium, or low. Deliver the result in a JSON blob where 'Priority' maps to the priority, 'Message' maps to the message for the bystander without questions, and 'Questions' maps to a python list of the questions you have."
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  # Adjust to "gpt-4" if needed
            messages=[
                {"role": "system", "content": detailed_prompt},
                {"role": "user", "content": instructions}
            ],
            temperature=0  # Lower temperature for more deterministic output
        )
        return response.choices[0].message.content

pinecone_api_key = "7623f706-02e2-427e-8e10-c1b77db64b56"
open_ai_api_key = "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"
assistant = EmergencyAssistanceLLM(pinecone_api_key, open_ai_api_key)
print(assistant.generate_response("I think I got stung by a jellyfish while swimming in the ocean."))