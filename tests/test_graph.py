"""End-to-end graph tests (offline 'fake' LLM provider)."""

from hr_policy_agent import HRPolicyAgent
from hr_policy_agent.config import Settings
from hr_policy_agent.feedback import handle_feedback


def _agent():
    return HRPolicyAgent(Settings(llm_provider="fake"))


def test_greeting_path():
    out = _agent().answer("hello")
    assert "How can I help" in out


def test_english_policy_path_has_answer_and_sources():
    state = _agent().run("How many PL days do I accrue?")
    assert state["language"] == "EN"
    assert "Personal Leave" in state["final_response"]
    assert "<summary><b>Source" in state["final_response"]  # citation card rendered


def test_spanish_policy_path():
    state = _agent().run("¿Cuántos días de vacaciones tengo?")
    assert state["language"] == "ES"
    assert "Manual del Miembro del Equipo" in state["final_response"]


def test_chat_store_logged():
    agent = _agent()
    agent.run("When is my pay date?")
    logged = agent.services.chat_store.logged
    assert logged and logged[-1]["operation"] == "AgentChatStore"
    assert logged[-1]["payload"]["userQuery"] == "When is my pay date?"


def test_tm_type_classification_flows_into_state():
    state = _agent().run("What is the attendance policy?")
    person = state["nodes"]["RETRIEVE_PERSON_DETAILS_SCRIPT"]["result"]
    assert person["tmType"] in ("Non Represented", "Represented", "Tavern")


def test_feedback_positive_and_negative():
    agent = _agent()
    settings = agent.settings
    store = agent.services.chat_store
    pos = handle_feedback("APPROVED", None, "EN", store, settings)
    assert not pos["needs_detail"] and "Thank you" in pos["response"]
    # Rejection without a valid reason asks again, then accepts "1".
    ask = handle_feedback("REJECTED", None, "EN", store, settings)
    assert ask["needs_detail"]
    done = handle_feedback("REJECTED", "1", "EN", store, settings)
    assert not done["needs_detail"]
    assert any(r["operation"] == "NegativeFeedback" for r in store.logged)


def test_cli_html_to_text_strips_markup():
    from hr_policy_agent.cli import html_to_text
    html = ('Answer line one.<br><br> <details open><summary><b>Source</b></summary>'
            '<div><span>1</span><span>Handbook</span><div>&ldquo;cited text&rdquo;</div></div></details><hr>')
    out = html_to_text(html)
    assert "<" not in out and ">" not in out          # no tags
    assert "Answer line one." in out
    assert "Source:" in out
    assert "1 Handbook" in out                          # badge + title not fused
