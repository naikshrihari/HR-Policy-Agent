"""LangGraph StateGraph for the HR Policy Workflow Agent.

This wires the ported nodes into the same control flow as the Oracle data pipeline.
The original graph mirrors an English and a Spanish copy of every node; because the
two copies are semantically identical, this port carries ``language`` in state and runs
one language-aware pipeline, choosing Spanish prompts/messages/RAG tools when the
Intent Router detects ES.  The observable behaviour matches the original flow:

    START
      -> Get User Session (ORA_USER_SESSION_TOOL)
      -> Fetch Worker Details (HCM getWorkerDetails)
      -> Encrypt Person Number
      -> Retrieve Person Details (classify Tavern / Represented / Non Represented)
      -> Input User Query (LLM)
      -> Intent Route (LLM: intent / tmType / language)
      -> GREETING?  yes -> greeting response (EN/ES) -> END
                    no  -> Create Unique Query (LLM) -> Combine Query
                        -> POLICY?  yes -> RAG (by tmType+language) -> Answer Agent
                                    no  -> Redirect (LLM)          -> Answer Agent
                        -> Final Answer Generator (LLM) -> Get Best Answer
                        -> Fetch System Mapping (referral)
                        -> (POLICY) citation selection + citation cards
                        -> Topic Classification -> HR Routing -> Agent Chat Store
                        -> compose final response
                        -> (optional) human feedback loop
                        -> END

Node outputs are stored in ``state["nodes"][CODE]`` exactly as the workflow addressed
them.  CODE-node outputs are wrapped in the Oracle ``{"result": ...}`` envelope so the
``$output.result`` template references resolve unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph

from .codenodes import citation, citation_details, person, system_mapping, topic_classifier, transforms
from .config import Settings
from .llm_nodes import run_llm_node
from .services.chat_store import ChatStoreClient
from .services.hcm import HCMClient
from .services.rag import BaseDocumentTool, build_document_tools
from .services.session import UserSessionTool
from .state import AgentState

GREETING_RESPONSES = {
    "EN": "Hello! How can I help you with HR policy questions today?",
    "ES": "¡Hola! ¿Cómo puedo ayudarte hoy con preguntas sobre la política de recursos humanos?",
}
TAVERN_SPANISH_RETURN = (
    "HR policy questions for Tavern Team Members are currently supported in English only. "
    "Please re-submit your question in English so I can provide the correct policy information."
)

# tmType -> RAG node code, per language.
_RAG_BY_TYPE = {
    "EN": {"TAVERN": "TAVERN_RAG", "REPRESENTED": "REPRESENTED_RAG",
           "NON REPRESENTED": "NON_REPRESENTED_RAG"},
    "ES": {"REPRESENTED": "REPRESENTED_RAG_SPANISH",
           "NON REPRESENTED": "NON_REPRESENTED_RAG_SPANISH"},
}


class Services:
    """Container for the external service clients used by graph nodes."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.session = UserSessionTool(settings.default_person_number)
        self.hcm = HCMClient(settings)
        self.chat_store = ChatStoreClient(settings)
        self.doc_tools: Dict[str, BaseDocumentTool] = build_document_tools(settings)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _set(code: str, value: Any) -> Dict[str, Any]:
    return {"nodes": {code: value}}


def _code_result(state: AgentState, code: str, default: Any = None) -> Any:
    node = (state.get("nodes") or {}).get(code) or {}
    return node.get("result", default) if isinstance(node, dict) else default


def _lang(state: AgentState) -> str:
    return "ES" if str(state.get("language", "EN")).upper() == "ES" else "EN"


def _intent(state: AgentState) -> str:
    route = (state.get("nodes") or {}).get("INTENT_ROUTE_LLM") or {}
    return route.get("intent", "POLICY")


def _tm_type(state: AgentState) -> str:
    route = (state.get("nodes") or {}).get("INTENT_ROUTE_LLM") or {}
    return str(route.get("tmType", "Non Represented")).upper()


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------
def build_graph(settings: Optional[Settings] = None, services: Optional[Services] = None,
                checkpointer: Any = None):
    settings = settings or Settings()
    services = services or Services(settings)

    # ---- Identity / classification ----
    def get_user_session(state: AgentState) -> Dict[str, Any]:
        return _set("GET_USER_SESSION", services.session.get_session(state.get("person_number")))

    def fetch_worker_details(state: AgentState) -> Dict[str, Any]:
        items = (state["nodes"]["GET_USER_SESSION"].get("items") or [{}])
        pn = items[0].get("PersonNumber") if items else None
        return _set("TM_CLASSIFCATION", services.hcm.get_worker_details(pn))

    def encrypt_person_number(state: AgentState) -> Dict[str, Any]:
        items = (state["nodes"]["GET_USER_SESSION"].get("items") or [{}])
        pn = items[0].get("PersonNumber") if items else None
        return _set("ENCRYPT_PERSON_NUMBER", {"result": person.encrypt_person_number(pn)})

    def retrieve_person_details(state: AgentState) -> Dict[str, Any]:
        hcm_data = state["nodes"]["TM_CLASSIFCATION"]
        return _set("RETRIEVE_PERSON_DETAILS_SCRIPT", {"result": person.retrieve_person_details(hcm_data)})

    # ---- Query understanding / routing ----
    def input_user_query(state: AgentState) -> Dict[str, Any]:
        return _set("INPUT_USER_QUERY", run_llm_node("INPUT_USER_QUERY", state, settings))

    def intent_route(state: AgentState) -> Dict[str, Any]:
        out = run_llm_node("INTENT_ROUTE_LLM", state, settings)
        language = "ES" if str(out.get("language", "EN")).upper() == "ES" else "EN"
        # LANGUAGE_ROUTER_ is a routing node in the original; expose its decision (the
        # language) so prompts that read {{...LANGUAGE_ROUTER_.$output}} resolve.
        return {"nodes": {"INTENT_ROUTE_LLM": out, "LANGUAGE_ROUTER_": language}, "language": language}

    def greeting_response(state: AgentState) -> Dict[str, Any]:
        return {"final_response": GREETING_RESPONSES[_lang(state)]}

    def create_unique_query(state: AgentState) -> Dict[str, Any]:
        code = "CREATE_UNIQUE_QUERY_SPANISH" if _lang(state) == "ES" else "CREATE_UNIQUE_QUERY"
        query = run_llm_node(code, state, settings)
        # QUERY_FORMULATION is the reformulated query alias some prompts read.
        return {"nodes": {code: query, "QUERY_FORMULATION": query}}

    def combine_query(state: AgentState) -> Dict[str, Any]:
        lang = _lang(state)
        cq_code = "CREATE_UNIQUE_QUERY_SPANISH" if lang == "ES" else "CREATE_UNIQUE_QUERY"
        combine_code = ("COMBINE_USER_QUERY_AND_QUERY_FORMULATION_CODE_SPANISH"
                        if lang == "ES" else "COMBINE_USER_QUERY_AND_QUERY_FORMULATION_CODE")
        query = (state["nodes"].get(cq_code) or "")
        constraints = (state["nodes"].get("INPUT_USER_QUERY") or {}).get("preservedConstraints", "")
        value = transforms.combine_user_query(query, constraints, lang)
        return _set(combine_code, {"result": value})

    # ---- Retrieval / answering ----
    def tavern_spanish_return(state: AgentState) -> Dict[str, Any]:
        return {"final_response": TAVERN_SPANISH_RETURN}

    def rag(state: AgentState) -> Dict[str, Any]:
        lang = _lang(state)
        tm = _tm_type(state)
        rag_code = _RAG_BY_TYPE[lang].get(tm) or _RAG_BY_TYPE[lang]["NON REPRESENTED"]
        # Retrieve with the CLEAN reformulated question. The Oracle COMBINE node wraps the
        # query with instruction boilerplate ("Question: … IMPORTANT: Retrieve only …")
        # meant for Oracle's RAG tool; that text pollutes a vector/keyword query and hurts
        # retrieval, so we search with the plain question (matching scripts.search).
        cq_code = "CREATE_UNIQUE_QUERY_SPANISH" if lang == "ES" else "CREATE_UNIQUE_QUERY"
        question = (state["nodes"].get(cq_code)
                    or (state["nodes"].get("INPUT_USER_QUERY") or {}).get("searchQuery")
                    or state.get("input_message", ""))
        result = services.doc_tools[rag_code].query(question)
        return _set(rag_code, result)

    def redirect(state: AgentState) -> Dict[str, Any]:
        code = "REDIRECT_LLM_SPANISH" if _lang(state) == "ES" else "REDIRECT_LLM"
        return _set(code, run_llm_node(code, state, settings))

    def answer_agent(state: AgentState) -> Dict[str, Any]:
        code = "ANSWER_AGENT_SPANISH" if _lang(state) == "ES" else "ANSWER_AGENT_"
        return _set(code, run_llm_node(code, state, settings))

    def final_answer(state: AgentState) -> Dict[str, Any]:
        code = "FINAL_ANWER_GENERATOR_SPANISH" if _lang(state) == "ES" else "FINAL_ANWER_GENERATOR"
        return _set(code, run_llm_node(code, state, settings))

    def best_answer(state: AgentState) -> Dict[str, Any]:
        lang = _lang(state)
        final_code = "FINAL_ANWER_GENERATOR_SPANISH" if lang == "ES" else "FINAL_ANWER_GENERATOR"
        answer_code = "ANSWER_AGENT_SPANISH" if lang == "ES" else "ANSWER_AGENT_"
        best_code = "GET_THE_BEST_ANSWER_SPANISH" if lang == "ES" else "GET_THE_BEST_ANSWER"
        value = transforms.get_the_best_answer(
            state["nodes"].get(final_code, ""), state["nodes"].get(answer_code, ""), lang)
        return _set(best_code, {"result": value})

    # ---- Referral system mapping ----
    def fetch_system_mapping(state: AgentState) -> Dict[str, Any]:
        lang = _lang(state)
        answer_code = "ANSWER_AGENT_SPANISH" if lang == "ES" else "ANSWER_AGENT_"
        map_code = ("FETCH_SYSTEM_MAPPING_SCRIPT_SPANISH" if lang == "ES"
                    else "FETCH_SYSTEM_MAPPING_SCRIPT")
        intent = _intent(state)
        if intent == "POLICY":
            rag_code = _RAG_BY_TYPE[lang].get(_tm_type(state)) or _RAG_BY_TYPE[lang]["NON REPRESENTED"]
            rag_value = (state["nodes"].get(rag_code) or {}).get("value", "")
        else:
            redirect_code = "REDIRECT_LLM_SPANISH" if lang == "ES" else "REDIRECT_LLM"
            rag_value = state["nodes"].get(redirect_code, "")
        result = system_mapping.fetch_system_mapping(
            agent_answer=state["nodes"].get(answer_code, ""),
            raw_question=state.get("input_message", ""),
            tm_type=_tm_type(state),
            query_intent=intent,
            rag_value=rag_value,
            language=lang,
        )
        return _set(map_code, {"result": result})

    # ---- Citation handling (POLICY only) ----
    def citation_flow(state: AgentState) -> Dict[str, Any]:
        lang = _lang(state)
        updates: Dict[str, Any] = {"nodes": {}}

        # RETURN_CITATION_SCRIPT — deterministic best-citation selection.
        rag_outputs = [state["nodes"].get(c) for c in (
            "NON_REPRESENTED_RAG", "REPRESENTED_RAG", "TAVERN_RAG",
            "NON_REPRESENTED_RAG_SPANISH", "REPRESENTED_RAG_SPANISH")]
        script_code = "RETURN_CITATION_SCRIPT_SPANISH" if lang == "ES" else "RETURN_CITATION_SCRIPT"
        script_json = citation.return_citation_script(rag_outputs)
        updates["nodes"][script_code] = {"result": script_json}

        import json
        details = json.loads(script_json)
        citation_only = None
        if len(details.get("Citation_Details", [])) == 0:
            only_code = ("GET_THE_RELEVANT_CITATION_ONLY_SPANISH" if lang == "ES"
                         else "GET_THE_RELEVANT_CITATION_ONLY")
            citation_only = run_llm_node(only_code, {**state, "nodes": {**state["nodes"], **updates["nodes"]}}, settings)
            updates["nodes"][only_code] = citation_only

        answer_code = "ANSWER_AGENT_SPANISH" if lang == "ES" else "ANSWER_AGENT_"
        resp_code = "RETURN_AGENT_RESPONSE_SPANISH" if lang == "ES" else "RETURN_AGENT_RESPONSE"
        agent_response = citation.return_agent_response(
            {"result": script_json}, citation_only, state["nodes"].get(answer_code, ""))
        updates["nodes"][resp_code] = {"result": agent_response}

        # GET_THE_CITATION_DETAILS — render the citation cards.
        rag_code = _RAG_BY_TYPE[lang].get(_tm_type(state)) or _RAG_BY_TYPE[lang]["NON REPRESENTED"]
        details_code = "GET_THE_CITATION_DETAILS_SPANISH" if lang == "ES" else "GET_THE_CITATION_DETAILS_ENGLISH"
        html = citation_details.get_citation_details(
            agent_response=agent_response,
            agent_response_topic=state["nodes"].get(answer_code, ""),
            rag_outputs=[state["nodes"].get(rag_code) or {}],
            query_text=state.get("input_message", ""),
            language=lang,
        )
        updates["nodes"][details_code] = {"result": html}
        return updates

    # ---- Topic classification / HR routing / logging ----
    def topic_classification(state: AgentState) -> Dict[str, Any]:
        lang = _lang(state)
        answer_code = "ANSWER_AGENT_SPANISH" if lang == "ES" else "ANSWER_AGENT_"
        code = "TOPIC_CLASSIFICATION_SCRIPT_SPANISH" if lang == "ES" else "TOPIC_CLASSIFICATION_SCRIPT"
        iuq = (state["nodes"].get("INPUT_USER_QUERY") or {}).get("searchQuery", state.get("input_message", ""))
        result = topic_classifier.classify_topic(iuq, state["nodes"].get(answer_code, ""), lang)
        return _set(code, {"result": result})

    def hr_routing(state: AgentState) -> Dict[str, Any]:
        lang = _lang(state)
        topic_code = "TOPIC_CLASSIFICATION_SCRIPT_SPANISH" if lang == "ES" else "TOPIC_CLASSIFICATION_SCRIPT"
        code = "HR_ROUTING_CLASSIFICATION_SPANISH" if lang == "ES" else "HR_ROUTING_CLASSIFICATION"
        topic = _code_result(state, topic_code, {}).get("topic_matched", "")
        result = transforms.hr_routing_classification(topic)
        return {"nodes": {code: {"result": result}}, "routed_to_hr": result["hr_routing"] == "1"}

    def agent_chat_store(state: AgentState) -> Dict[str, Any]:
        lang = _lang(state)
        person_d = _code_result(state, "RETRIEVE_PERSON_DETAILS_SCRIPT", {})
        hr_code = "HR_ROUTING_CLASSIFICATION_SPANISH" if lang == "ES" else "HR_ROUTING_CLASSIFICATION"
        topic_code = "TOPIC_CLASSIFICATION_SCRIPT_SPANISH" if lang == "ES" else "TOPIC_CLASSIFICATION_SCRIPT"
        best_code = "GET_THE_BEST_ANSWER_SPANISH" if lang == "ES" else "GET_THE_BEST_ANSWER"
        payload = {
            "hrRouting": _code_result(state, hr_code, {}).get("hr_routing"),
            "location_name": person_d.get("locationName"),
            "person_number_hash": _code_result(state, "ENCRYPT_PERSON_NUMBER", {}).get("person_number_hash"),
            "session_id": state.get("conversation_id", ""),
            "conversation_turn_id": state.get("trace_id", ""),
            "tmType": person_d.get("tmType"),
            "tmDepartment": person_d.get("tmDepartment"),
            "tmProperty": person_d.get("tmProperty"),
            "userLanguage": lang,
            "userQuery": state.get("input_message", ""),
            "topic_matched": _code_result(state, topic_code, {}).get("topic_matched"),
            "full_agent_response": _code_result(state, best_code, ""),
        }
        services.chat_store.agent_chat_store(payload)
        return {}

    def compose_response(state: AgentState) -> Dict[str, Any]:
        lang = _lang(state)
        best_code = "GET_THE_BEST_ANSWER_SPANISH" if lang == "ES" else "GET_THE_BEST_ANSWER"
        map_code = "FETCH_SYSTEM_MAPPING_SCRIPT_SPANISH" if lang == "ES" else "FETCH_SYSTEM_MAPPING_SCRIPT"
        details_code = "GET_THE_CITATION_DETAILS_SPANISH" if lang == "ES" else "GET_THE_CITATION_DETAILS_ENGLISH"
        best = _code_result(state, best_code, "") or ""
        referral = (_code_result(state, map_code, {}) or {}).get("referral_message", "")
        citations = _code_result(state, details_code, "") or ""
        # Mirrors the FEEDBACK node's message template.
        final = f"{best}\n\n<br><br> {referral}\n\n {citations} ".rstrip()
        return {"final_response": final}

    # ---- Routers ----
    def route_after_intent(state: AgentState) -> str:
        return "greeting" if _intent(state) == "GREETING" else "create_unique_query"

    def route_after_combine(state: AgentState) -> str:
        # Spanish + Tavern is redirected to an English-only notice.
        if _lang(state) == "ES" and _tm_type(state) == "TAVERN":
            return "tavern_spanish_return"
        return "rag" if _intent(state) == "POLICY" else "redirect"

    def route_after_system_mapping(state: AgentState) -> str:
        return "citation_flow" if _intent(state) == "POLICY" else "topic_classification"

    # ---- Assemble ----
    g = StateGraph(AgentState)
    g.add_node("get_user_session", get_user_session)
    g.add_node("fetch_worker_details", fetch_worker_details)
    g.add_node("encrypt_person_number", encrypt_person_number)
    g.add_node("retrieve_person_details", retrieve_person_details)
    g.add_node("input_user_query", input_user_query)
    g.add_node("intent_route", intent_route)
    g.add_node("greeting", greeting_response)
    g.add_node("create_unique_query", create_unique_query)
    g.add_node("combine_query", combine_query)
    g.add_node("tavern_spanish_return", tavern_spanish_return)
    g.add_node("rag", rag)
    g.add_node("redirect", redirect)
    g.add_node("answer_agent", answer_agent)
    g.add_node("final_answer", final_answer)
    g.add_node("best_answer", best_answer)
    g.add_node("fetch_system_mapping", fetch_system_mapping)
    g.add_node("citation_flow", citation_flow)
    g.add_node("topic_classification", topic_classification)
    g.add_node("hr_routing", hr_routing)
    g.add_node("agent_chat_store", agent_chat_store)
    g.add_node("compose_response", compose_response)

    g.add_edge(START, "get_user_session")
    g.add_edge("get_user_session", "fetch_worker_details")
    g.add_edge("fetch_worker_details", "encrypt_person_number")
    g.add_edge("encrypt_person_number", "retrieve_person_details")
    g.add_edge("retrieve_person_details", "input_user_query")
    g.add_edge("input_user_query", "intent_route")
    g.add_conditional_edges("intent_route", route_after_intent,
                            {"greeting": "greeting", "create_unique_query": "create_unique_query"})
    g.add_edge("greeting", END)
    g.add_edge("create_unique_query", "combine_query")
    g.add_conditional_edges("combine_query", route_after_combine,
                            {"tavern_spanish_return": "tavern_spanish_return",
                             "rag": "rag", "redirect": "redirect"})
    g.add_edge("tavern_spanish_return", END)
    g.add_edge("rag", "answer_agent")
    g.add_edge("redirect", "answer_agent")
    g.add_edge("answer_agent", "final_answer")
    g.add_edge("final_answer", "best_answer")
    g.add_edge("best_answer", "fetch_system_mapping")
    g.add_conditional_edges("fetch_system_mapping", route_after_system_mapping,
                            {"citation_flow": "citation_flow", "topic_classification": "topic_classification"})
    g.add_edge("citation_flow", "topic_classification")
    g.add_edge("topic_classification", "hr_routing")
    g.add_edge("hr_routing", "agent_chat_store")
    g.add_edge("agent_chat_store", "compose_response")
    g.add_edge("compose_response", END)

    return g.compile(checkpointer=checkpointer)
