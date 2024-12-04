import openai
from openai import OpenAI
import json
from trulens.apps.custom import instrument

class BaselineLLM:
    def __init__(self, open_ai_api_key):
        self.client = OpenAI(api_key=open_ai_api_key)
        self.simple_state = []
        self.detailed_state = []

    @instrument
    def get_complete_history(self, bystander_transcript, simple):
        if simple is True:
            past_updates = [state for state, _ in self.simple_state]
        else:
            past_updates = [state for state, _ in self.detailed_state]
        history =  " ".join(past_updates)
        #return f"My most recent update is that: {bystander_transcript}. My past updates were: {history}"
        return f"{history}. {bystander_transcript}."

    def postprocess_response(self, response_jsonified):
            if "{" not in response_jsonified:
                return response_jsonified
            first_bracket = response_jsonified.find("{")
            second_bracket = response_jsonified.rfind("}")
            response_json = json.loads(response_jsonified[first_bracket:second_bracket + 1])
            all_questions = " ".join(response_json["Questions"])
            message = response_json["Message"] + all_questions
            return message
    
    @instrument
    def generate_baseline_response(self, bystander_transcript):
        complete_history = self.get_complete_history(bystander_transcript, simple=True)
        # print(complete_history)
        # Generate a response using the LLM model with the bystander transcript and context
        past_interactions = [f"In response to - {transcript} - the dispatcher said: {response}" for transcript, response in self.simple_state]
        history_prompt = f"I will provide a history of past interactions between the dispatcher and the bystander. You should make sure you are providing new and relevant insights to the dispatcher as you are updated. However, you should make sure you responses are taking into account all historical context and how new updates fit into what you already understand about the situation. Here is the history:" + " ".join(past_interactions)
        prompt = f"You are a dispatcher assistant. You will receive what a bystander says about an emergency along with some context. Your goal is to provide instructions for what the dispatcher should tell the bystander. {history_prompt}"
        instructions =f"Bystander says: {bystander_transcript}\n" \
                 "Provide clear, step-by-step instructions for the dispatcher to relay to the bystander. Include any clarifying questions that the dispatcher should ask the bystander to better inform response." \
                 "Inform the dispatcher whether the priority is high, medium, or low. Deliver the result in a JSON blob where \"Priority\" maps to the priority, \"Message\" maps to the message for the bystander without questions, and \"Questions\" maps to the questions you have."
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  # Adjust to "gpt-4" if needed
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": instructions}
            ],
            temperature=0  # Lower temperature for more deterministic output
        )
        response_text = response.choices[0].message.content
        # final_response = self.postprocess_response(response_text)
        # self.simple_state.append((bystander_transcript, final_response))
        # return self.postprocess_response(response_text)
        return response_text
    
    @instrument
    def generate_detailed_baseline_response(self, bystander_transcript):
        complete_history = self.get_complete_history(bystander_transcript, simple=False)
        # print(complete_history)
        past_interactions = [f"In response to - {transcript} - the dispatcher said: {response}" for transcript, response in self.detailed_state]
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
        instructions =f"Bystander says: {bystander_transcript}\n" \
                 "Provide clear, step-by-step instructions for the dispatcher to relay to the bystander. Include any clarifying questions that the dispatcher should ask the bystander to better inform response." \
                 "Inform the dispatcher whether the priority is high, medium, or low. Deliver the result in a JSON blob where 'Priority' maps to the priority, 'Message' maps to the message for the bystander without questions, and 'Questions' maps to the questions you have."
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  # Adjust to "gpt-4" if needed
            messages=[
                {"role": "system", "content": detailed_prompt},
                {"role": "user", "content": instructions}
            ],
            temperature=0  # Lower temperature for more deterministic output
        )
        response_text = response.choices[0].message.content
        # final_response = self.postprocess_response(response_text)
        # self.detailed_state.append((bystander_transcript, final_response))
        # return self.postprocess_response(response_text)
        return response_text

# open_ai_api_key = ""
# assistant = BaselineLLM(open_ai_api_key)
# print(assistant.generate_baseline_response("I am seeing someone unconscious on the ground, bleeding profusely from the head.")) 
# print(assistant.generate_detailed_baseline_response("I am seeing someone unconscious on the ground, bleeding profusely from the head."))