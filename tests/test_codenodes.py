"""Unit tests for the ported JS CODE nodes."""

import json

from hr_policy_agent.codenodes import citation, person, system_mapping, topic_classifier, transforms


# --------------------------------------------------------------------------- person
def test_encrypt_person_number_fallback_and_hash():
    # Null falls back to 123456789 and is hashed to a 15-char uppercase hex.
    out = person.encrypt_person_number(None)
    assert len(out["person_number_hash"]) == 15
    assert out["person_number_hash"] == out["person_number_hash"].upper()
    # A 15-char value passes through unchanged.
    assert person.encrypt_person_number("A" * 15)["person_number_hash"] == "A" * 15


def _hcm(legal="Station Casinos LLC", bu="N/A", status="ACTIVE", primary=True, atype="E"):
    return {"items": [{"workRelationships": [{"LegalEmployerName": legal, "assignments": [
        {"AssignmentStatusType": status, "PrimaryFlag": primary, "AssignmentType": atype,
         "DepartmentName": "F&B", "BargainingUnitCode": bu, "LocationName": "Red Rock"}]}]}]}


def test_retrieve_person_details_classification():
    assert person.retrieve_person_details(_hcm())["tmType"] == "Non Represented"
    assert person.retrieve_person_details(_hcm(bu="CULINARY-226"))["tmType"] == "Represented"
    assert person.retrieve_person_details(_hcm(legal="SCT Nevada LLC"))["tmType"] == "Tavern"
    # No active/primary assignment -> default fallback.
    assert person.retrieve_person_details(_hcm(status="INACTIVE"))["tmType"] == "Non Represented"
    assert person.retrieve_person_details({})["tmType"] == "Non Represented"


# --------------------------------------------------------------------------- transforms
def test_combine_user_query_language_and_constraint():
    assert "Question:" in transforms.combine_user_query("q", "", "EN")
    assert "CONSTRAINT:" in transforms.combine_user_query("q", "full-time only", "EN")
    assert "Pregunta:" in transforms.combine_user_query("q", "", "ES")


def test_get_best_answer_not_covered_fallback():
    final = "This topic is not covered by the handbook."
    fallback = "Here is the real answer."
    assert transforms.get_the_best_answer(final, fallback, "EN") == fallback
    assert transforms.get_the_best_answer("A clear answer.", fallback, "EN") == "A clear answer."


def test_hr_routing_classification():
    assert transforms.hr_routing_classification("harassment_report")["hr_routing"] == "1"
    assert transforms.hr_routing_classification("policy_inquiry")["hr_routing"] == "0"


# --------------------------------------------------------------------------- topic classifier
def test_topic_classifier_examples():
    assert topic_classifier.classify_topic("I want to hurt myself", "", "EN")["topic_matched"] == "CRISIS"
    assert topic_classifier.classify_topic("When is my pay date?", "", "EN")["topic_matched"] == "POLICY_INQUIRY"
    assert topic_classifier.classify_topic("", "", "EN")["topic_matched"] == "OFF_TOPIC"
    # Spanish route.
    assert topic_classifier.classify_topic("¿Cuándo es el día de pago?", "", "ES")["topic_matched"] == "POLICY_INQUIRY"


# --------------------------------------------------------------------------- system mapping
def test_system_mapping_personal_pay_routes_to_adp():
    out = system_mapping.fetch_system_mapping(
        agent_answer="", raw_question="Where can I see my pay stub?",
        tm_type="NON REPRESENTED", query_intent="POLICY", rag_value="", language="EN")
    assert out["question_type"] == "PERSONAL"
    assert out["system_mapping"] == ["ADP"]
    assert "ADP" in out["referral_message"]


def test_system_mapping_non_personal_is_policy():
    out = system_mapping.fetch_system_mapping(
        agent_answer="", raw_question="What is the bereavement policy?",
        tm_type="NON REPRESENTED", query_intent="POLICY", rag_value="", language="EN")
    assert out["question_type"] == "POLICY"


# --------------------------------------------------------------------------- citation
def test_return_citation_script_selects_matching_chunk():
    rag = [{"value": "Team Members accrue personal leave based on length of service and hours.",
            "citations": [
                {"citedText": "Team Members accrue personal leave based on length of service and hours worked.",
                 "documentIdentificationCriteria": {"documentTitle": "Handbook"}},
                {"citedText": "Unrelated section about parking permits and garages.",
                 "documentIdentificationCriteria": {"documentTitle": "Handbook"}},
            ]}]
    out = json.loads(citation.return_citation_script(rag))
    assert out["Document_Title"] == ["Handbook"]
    assert "personal leave" in out["Citation_Details"][0]


def test_return_agent_response_priority():
    script = {"result": json.dumps({"Document_Title": ["H"], "Citation_Details": ["cited answer"]})}
    assert citation.return_agent_response(script, None, {"value": "x"}) == "cited answer"
    empty = {"result": json.dumps({"Document_Title": [], "Citation_Details": []})}
    assert citation.return_agent_response(empty, None, {"value": "fallback answer"}) == "fallback answer"
