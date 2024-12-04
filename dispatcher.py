from llm_module import EmergencyAssistanceLLM
from baseline import BaselineLLM
from rag_module import RAGQueryModule
import json
from feedback_evals import Feedback_Evals
import pandas as pd
from trulens.apps.custom import instrument

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
    
    @instrument
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
    def postprocess_plan(self, response):
        questions = response["Questions"]
        next_steps = response["Next Steps"]
        questions_str = " ".join(questions)
        if not questions:
            llm_message = f"At this point, I seem to have a good understanding of the situation. Please proceed with the following next steps: {next_steps}"
        else:
            llm_message = f"Here are some pertinent clarifying questions to ask: {questions_str} Contingent upon those answers, here are potential next steps: {next_steps}"
        return llm_message

    @instrument
    def query_app(self, bystander_transcript, do_eval=True, with_planning=False, one_shot=False):
        """
        Parameters:
        - bystander_transcript: The transcript of what the bystander says on the phone.
        
        Returns:
        - response: A response to guide the dispatcher.
        """
        full_transcript = self.llm_app.get_complete_history(bystander_transcript)
        llm_response = self.llm_app.generate_response(bystander_transcript, with_planning)
        if with_planning:
            llm_message = self.postprocess_plan(llm_response)
        else:
            llm_message = self.postprocess_response(llm_response)
        if not one_shot:
            self.llm_app.state.append((bystander_transcript, llm_message))
        if not do_eval:
            return llm_message
        simple_baseline_response = self.baseline_app.generate_baseline_response(bystander_transcript)
        simple_baseline_message = self.postprocess_response(simple_baseline_response)
        self.baseline_app.simple_state.append((bystander_transcript, simple_baseline_message))

        detailed_baseline_response = self.baseline_app.generate_detailed_baseline_response(bystander_transcript)
        detailed_baseline_message = self.postprocess_response(detailed_baseline_response)
        self.baseline_app.detailed_state.append((bystander_transcript, detailed_baseline_message))
        
        context = self.feedback_evals.get_context(full_transcript)
        # determine context applicability? 
        print(f"dispatcher/full_transcript: {full_transcript}\n")
        # print(f"dispatcher/unprocessed_message: {llm_response}\n")
        # print(f"dispatcher/postprocessed_plan: {llm_message}\n")
        print(f"dispatcher/context: {context}\n")
        llm_scores = self.feedback_evals.evaluate_response(context, full_transcript, llm_message)
        simple_baseline_scores = self.feedback_evals.evaluate_post_hoc_grounding(full_transcript, simple_baseline_message)
        detailed_baseline_scores = self.feedback_evals.evaluate_post_hoc_grounding(full_transcript, detailed_baseline_message)

        # Log evaluation
        evaluation_entry = {
            "transcript": full_transcript,
            "context": context,
            "llm_message": llm_message,
            "baseline_context": self.feedback_evals.get_context(simple_baseline_message),
            "baseline_message": simple_baseline_message,
            "detailed_baseline_context": self.feedback_evals.get_context(detailed_baseline_message),
            "detailed_baseline_message": detailed_baseline_message,
            "llm_scores": llm_scores,
            "baseline_scores": simple_baseline_scores,
            "detailed_baseline_scores": detailed_baseline_scores
        }
        self.feedback_evals.log_evaluation(evaluation_entry)

        return evaluation_entry
        #return llm_message, simple_baseline_message, detailed_baseline_message
    
    def run(self, do_eval=True, with_planning=True):
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
            evaluation = self.query_app(user_input, do_eval, with_planning)

            # Display responses and evaluations
            print("\n--- Emergency Assistance LLM Response ---")
            print(evaluation["llm_message"])
            print("\nLLM Evaluation Scores:")
            print(self.feedback_evals.format_scores(evaluation["llm_scores"]))

            print("\n--- Simple Baseline Response ---")
            print(evaluation["baseline_message"])
            print("\nSimple Baseline Evaluation Scores:")
            print(self.feedback_evals.format_scores(evaluation["baseline_scores"]))

            print("\n--- Detailed Baseline Response ---")
            print(evaluation["detailed_baseline_message"])
            print("\nDetailed Baseline Evaluation Scores:")
            print(self.feedback_evals.format_scores(evaluation["detailed_baseline_scores"]))


pinecone_api_key = ""
open_ai_api_key = ""
# test_prompts = ["I came across someone who is bleeding profusely from their head.",
# "I found someone lying on the ground unconscious but breathing.",
# "There's a person choking and can't seem to breathe or talk.",
# "I see someone at the beach with bright red swelling on their leg and they're in a lot of pain.",
# "Someone is holding their chest and says they're having trouble breathing.",
# "I found a person who is shivering and their lips are turning blue after being outside in the cold.",
# "I see someone having a seizure in the park.",
# "Someone tripped and fell; now they can't move their leg, and it's at an odd angle.",
# "A person at the pool seems to be drowning and isn't coming up for air.",
# "I came across a cyclist who crashed and is bleeding heavily from their arm.",
# "There's a child who touched a hot stove and has a severe burn on their hand.",
# "A person fainted in the crowd and is now unresponsive.",
# "I found someone holding their lower abdomen and crying out in pain.",
# "Someone got stung by a bee and now their face is swelling rapidly.",
# "A person was hit by a car and they seem to have a head injury, but they're still conscious."]

# case_studies = {"Respiratory_Distress": "DISPATCH INFO: 15-year-old male with “chest pain”. SCENE INFO: On arrival you find a young man sitting on a park bench with friends. HISTORY: He is endorsing chest pain and difficulty breathing. His friends say his symptoms began after vaping marijuana purchased from an unlicensed dealer. He has vaped many times before without significant problems. ALLERGIES: None. MEDS: Denies medical history or medications. ASSESSMENT: Patient is mildly agitated, diaphoretic, and speaking in short sentences. He is breathing rapidly and appears to be tensing his neck muscles with breathing.", 
#                 "Pesticide": "Paramedics are called to the home of a 56-year-old male in respiratory distress after an intentional ingestion of an insect spray concentrate.  The patient called 9-1-1 after he started to have abdominal cramps and began to vomit profusely.  He then began to wheeze and have difficulty breathing.  Upon EMS arrival he is tearful and diaphoretic.  He has difficulty speaking due to his respiratory distress with wheezing and significant airway secretions noted.  A bottle of the insect spray concentrate is next to the patient with liquid on the floor and on his clothing.",
#                 "Emergency_Childbirth": "EMS responds to the home of a 34-year-old female complaining of sudden onset abdominal pressure. The patient states the pressure started two hours prior to calling 911 and is worsening. As paramedics attempt to help her from her bed, she experiences a gush of clear fluid from her vagina. She exclaims that she feels an intense urge to push. She states she cannot remember the date of her last menstrual period. Exam notes an obese female in moderate distress clutching her lower abdomen.",
#                 "Blast_Injury": "EMS responds to an explosion at an industrial site.  A 44-year-old male was performing maintenance work when a pressurized gas tank in the next room exploded.  The patient was thrown 15 feet, striking an adjacent wall.  Co-workers called 911 following the blast.  The patient is extricated from the blast site within minutes.",
#                 "Emergency_Minor": "EMS responds to a private home where a 15-year-old female complains of abdominal pain.  She is at her friend’s home.  The 16-year-old friend called 911 because the patient was “getting worse and worse”.  The patient provides little history, but the friend states that the patient has been at her home for the last day and has been complaining of abdominal pain the entire time.  On initial survey, the patient appears pale.  She is alert and oriented, cool to touch, and appears weak.  Her abdomen is diffusely tender and mildly distended.  There are no adults present.",
#                 "Seizure": "EMS responds to a local convenience store for a 24-year-old male having a seizure.  The patient presents lying on the floor supine and convulsing in a tonic-clonic motion. He is unresponsive to verbal commands, and there is blood oozing from his mouth. A friend who accompanied the patient states that both of them had been drinking all day while binge-watching old episodes of Game of Thrones.  When they arrived at the carryout to purchase more beer, the patient complained that the lights in the store were bothering his eyes.  Almost immediately after saying that, he suddenly collapsed.  The friend caught the patient and lowered him slowly down to the floor, where he immediately began shaking. The shaking has been ongoing now for over ten minutes.  According to the friend, similar events have occurred this past year since the patient was in a motorcycle accident and hurt his head.  Initial vital signs reveal ventilations 30/minute, shallow and gurgling; pulse 112/minute and regular; blood pressure is unobtainable; pulse oximetry 91 percent on room air; blood glucose 134 mg/dL; skin is diaphoretic and warm to the touch."}

# # csv_path = "case_studies.csv"
# # case_studies_df = pd.read_csv(csv_path)

# for case, prompt in case_studies.items():
#     app = DispatcherApp(pinecone_api_key, open_ai_api_key)
#     json_filename = f"unplanned_case_study_{case}.json"
#     evaluation = app.query_app(prompt, do_eval=True, with_planning=False)
#     app.feedback_evals.save_evaluations(json_filename)
#     print(f"Finished saving evaluations for {json_filename}")

# app = DispatcherApp(pinecone_api_key, open_ai_api_key)
# app.run(with_planning=True)
