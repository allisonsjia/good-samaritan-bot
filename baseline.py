import openai
from openai import OpenAI

class BaselineLLM:
    def __init__(self, open_ai_api_key):
        self.client = OpenAI(api_key=open_ai_api_key)

    def generate_baseline_response(self, bystander_transcript):
        prompt = f"You are a dispatcher assistant. You will receive what a bystander says about an emergency along with some context. Your goal is to provide instructions for what the dispatcher should tell the bystander."
        instructions =f"Bystander says: {bystander_transcript}\n" \
                 "Provide clear, step-by-step instructions for the dispatcher to relay to the bystander. Include any clarifying questions that the dispatcher should ask the bystander to better inform response." \
                 "Inform the dispatcher whether the priority is high, medium, or low. Deliver the result in a JSON blob where 'Priority' maps to the priority, 'Message' maps to the message for the bystander without questions, and 'Questions' maps to the questions you have."
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",  # Adjust to "gpt-4" if needed
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": instructions}
            ],
            temperature=0,
        )
        return response.choices[0].message.content
    
    def generate_detailed_baseline_response(self, bystander_transcript):
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
            Your goal is to provide instructions for what the dispatcher should tell the bystander.
                  """
        )
        instructions =f"Bystander says: {bystander_transcript}\n" \
                 "Provide clear, step-by-step instructions for the dispatcher to relay to the bystander. Include any clarifying questions that the dispatcher should ask the bystander to better inform response." \
                 "Inform the dispatcher whether the priority is high, medium, or low. Deliver the result in a JSON blob where 'Priority' maps to the priority, 'Message' maps to the message for the bystander without questions, and 'Questions' maps to the questions you have."
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",  # Adjust to "gpt-4" if needed
            messages=[
                {"role": "system", "content": detailed_prompt},
                {"role": "user", "content": instructions}
            ],
            temperature=0,
        )
        return response.choices[0].message.content

# open_ai_api_key = ""
# assistant = BaselineLLM(open_ai_api_key)
# print(assistant.generate_baseline_response("I am seeing someone unconscious on the ground, bleeding profusely from the head.")) 
# print(assistant.generate_detailed_baseline_response("I am seeing someone unconscious on the ground, bleeding profusely from the head."))