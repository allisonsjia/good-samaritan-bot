import json
from trulens.core import Feedback
from trulens.feedback import GroundTruthAgreement
from trulens.providers.openai import OpenAI
import pandas as pd
from dispatcher import DispatcherApp
from trulens.apps.custom import TruCustomApp
from trulens.core.session import TruSession
from trulens.dashboard import run_dashboard, stop_dashboard
import os
import openai
from baseline import BaselineLLM

session = TruSession()
session.reset_database()

pinecone_api_key = "7623f706-02e2-427e-8e10-c1b77db64b56"
open_ai_api_key = "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"


csv_path = "case_studies.csv"
case_studies_df = pd.read_csv(csv_path)

# Remove extraneous '\n' characters from the 'query' and 'expected_response' columns
case_studies_df['query'] = case_studies_df['query'].str.replace('\n', ' ', regex=False)
case_studies_df['expected_response'] = case_studies_df['expected_response'].str.replace('\n', ' ', regex=False)

os.environ["OPENAI_API_KEY"] = open_ai_api_key
openai.api_key = open_ai_api_key
provider = OpenAI()

golden_set = case_studies_df.to_dict(orient='records')
f_agreement = (Feedback(
    GroundTruthAgreement(golden_set, provider=OpenAI()).agreement_measure, name = "Agreement"
    ).on_input_output()
    )

#f_bert = (Feedback(GroundTruthAgreement(golden_set, provider=OpenAI()).bert_score, name= "BERT").on_input_output())

# emergency_responder = DispatcherApp(pinecone_api_key, open_ai_api_key)
# tru_emergency_responder = TruCustomApp(
#     emergency_responder,
#     app_name="Emergency Responder",
#     app_version="planned_llm_responder",
#     feedbacks=[f_agreement]
# )

# for query in case_studies_df['query']:
#     with tru_emergency_responder as recording:
#         try:
#             response = emergency_responder.query_app(query, do_eval=False, with_planning=True, one_shot=True)
#             #print(f"{query}: {response}\n")
#         except Exception as e:
#             continue
# print("finished planned")

# unplanned_emergency_responder = DispatcherApp(pinecone_api_key, open_ai_api_key)
# tru_emergency_responder_unplanned = TruCustomApp(
#     unplanned_emergency_responder,
#     app_name="Emergency Responder",
#     app_version="unplanned_llm_responder",
#     feedbacks=[f_agreement]
# )

# for query in case_studies_df['query']:
#     with tru_emergency_responder_unplanned as recording:
#         try:
#             response = unplanned_emergency_responder.query_app(query, do_eval=False, with_planning=False, one_shot=True)
#             #print(f"{query}: {response}\n")
#         except Exception as e:
#             continue
# print("finished unplanned")

baseline = BaselineLLM(open_ai_api_key)
tru_baseline_responder = TruCustomApp(
    baseline,
    app_name="Emergency Responder",
    app_version="Simple Baseline Responder",
    feedbacks=[f_agreement]
)

for query in case_studies_df['query']:
    with tru_baseline_responder as recording:
        try:
            response = baseline.generate_baseline_response(query)
            #print(f"{query}: {response}\n")
        except Exception as e:
            continue
print("finished simple baseline")

# detailed_baseline = BaselineLLM(open_ai_api_key)
# tru_detailed_baseline_responder = TruCustomApp(
#     detailed_baseline,
#     app_name="Emergency Responder",
#     app_version="Detailed Baseline Responder",
#     feedbacks=[f_agreement]
# )

# for query in case_studies_df['query']:
#     with tru_detailed_baseline_responder as recording:
#         try:
#             response = detailed_baseline.generate_detailed_baseline_response(query)
#             #print(f"{query}: {response}\n")
#         except Exception as e:
#             continue

# print("finished detailed baseline")

# for query in case_studies_df['query']:
#     try:
#         # Attempt to get a response from the emergency responder
#         response = emergency_responder.query_app(query, do_eval=False, with_planning=True, one_shot=True)
#         llm_responses[query] = response
#     except Exception as e:
#         continue

# print(llm_responses)

run_dashboard(session=session)