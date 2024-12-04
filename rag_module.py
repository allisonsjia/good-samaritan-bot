from pinecone import Pinecone
import openai
from openai import OpenAI
from trulens.apps.custom import instrument

INDEX_NAMES = [
    'airway-breathing',
    'bone-joint',
    'cardiovascular-cpr',
    'other-emergencies',
    'scene-management',
    'wounds-bandages',
    'basic-life-support'
]

class RAGQueryModule:
    def __init__(self, pinecone_api_key, open_ai_api_key, model="multilingual-e5-large"):
        # Initialize Pinecone and connect to the existing index
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.model = model
        self.client = OpenAI(api_key=open_ai_api_key)
    
        # Function to determine the index name using OpenAI LLM
    @instrument
    def get_index_name_from_query(self, query_text, candidate_labels=INDEX_NAMES):
        # Create a prompt to help classify the query text
        detailed_prompt = f"""
        You are an assistant that helps identify which category a given query belongs to. Here are the available categories: {', '.join(candidate_labels)}.
        - The 'airway-breathing' category includes information about hypoxia, effective and ineffective breathing, breathing and emergencies caused by illness, and choking.
        - The 'bone-joint' category includes information about fractures, dislocations, sprains, head and spinal injuries, pelvic injury, chest injury, splinting materials, first aid for specific bone & joint injuries, and strains.
        - The 'cardiovascular-cpr' category includes information about cardiovascular disease, angina and heart attack, chain of survival, stroke, transient ischemic attack, cardiac arrest, cardiopulmonary resuscitation (CPR), automated external defibrillation (AED).
        - The 'other-emergencies' category includes information about diabetes, seizures and convulsions, opioid overdose, environmental emergencies, cold-related injuries, heat-related injuries, poisoning, emergency childbirth and miscarriage, and mental health.
        - The 'scene-management' category includes information about steps of emergency scene management, scene survey, primary survey, secondary survey, ongoing casualty care, multiple casualty management (triage), lifting and moving, and extrication.
        - The 'wounds-bandages' category includes information about dressings, bandages, and slings, types of wounds, bleeding, internal bleeding, amputations, minor wound care, first aid for hand and foot injuries, chest injuries, abdominal injuries, crush injuries, scalp injuries, facial injuries, eye injuries, burns, bites, and stings.
        - The 'basic-life-support' category includes information about age categories for resuscitation, artificial respiration, cardiopulmonary resuscitation (CPR), and quick first aid reference.
        You must respond only with the name of the category.
        """

        # Call the OpenAI API to classify the query
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",  # Adjust to "gpt-4" if needed
            messages=[
                {"role": "system", "content": detailed_prompt},
                {"role": "user", "content": f"Given the following query, \"{query_text}\", decide which category best fits. Return only the best category name from the 7 possible categories: airway-breathing, bone-joint, cardiovascular-cpr, other-emergencies, scene-management, wounds-bandages, and basic-life-support."}
            ],
            max_tokens=20,
            temperature=0  # Lower temperature for more deterministic output
        )

        # Extract the best index name from the response
        best_index = response.choices[0].message.content.strip()
        for index_name in INDEX_NAMES:
            if index_name in best_index:
                best_index = index_name
                break

        #print(f"Determined index name: {best_index}")

        # Validate that the returned index is in the candidate list
        if best_index not in candidate_labels:
            raise ValueError("The LLM returned an index name that is not valid.")

        return best_index
    
        # Function to query Pinecone based on the determined index name
    @instrument
    def query_index_by_text(self, query_text):
        # Get the relevant index name using the LLM function
        index_name = self.get_index_name_from_query(query_text)

        # Connect to the chosen index
        index = self.pc.Index(index_name)

        # Generate embedding for the query text
        query_embedding = self.pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query_text],
        parameters={
            "input_type": "query"
        })

        # Perform the search
        results = index.query(
            namespace=f"{index_name}-namespace",
            vector=query_embedding[0].values,
            top_k=5,
            include_values=False,
            include_metadata=True
        )

        return results

        # Print search results
        # print("Query Results:")
        # for match in results["matches"]:
        #     print(f"Score: {match['score']}")
        #     print(f"Stimulus: {match['metadata']['stimulus']}")
        #     print(f"Instructions: {match['metadata']['instructions']}\n")


# pinecone_api_key = ""
# open_ai_api_key = ""
# rag_query_module = RAGQueryModule(pinecone_api_key=pinecone_api_key, open_ai_api_key=open_ai_api_key)

# query_text = "Please ensure your safety first. If you can do so safely, approach the person and check if they are conscious and responsive. If they are awake, calmly ask them their name and what happened. Apply gentle pressure to the bleeding area with a clean cloth or your hand to help control the bleeding. If the bleeding does not stop or if they become unresponsive, call for emergency services immediately. Do not move them unless there is an immediate danger. Stay with them and provide reassurance until help arrives.Is the person conscious and able to respond? Can you see the source of the bleeding clearly? Is there any visible danger in the area, such as broken glass or other hazards? Are there any other people around who can assist you? What is the person's approximate age and gender?"
# results = rag_query_module.query_index_by_text(query_text)
# for match in results["matches"]:
#     print(f"Score: {match['score']}")
#     print(f"Stimulus: {match['metadata']['stimulus']}")
#     print(f"Instructions: {match['metadata']['instructions']}\n")
