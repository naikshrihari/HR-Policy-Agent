"""Port of FETCH_SYSTEM_MAPPING_SCRIPT (+ Spanish) — v3.11.

Station Casinos "System Mapping Agent": a deterministic post-processor that decides,
for a PERSONAL question, which self-service system the Team Member should be referred
to (ADP, Benefits Connect, Absence Resources, Ask your Manager, HCM, Human Resources)
and produces the grammar-safe referral sentence appended to the answer.

The anchor phrase tables are bilingual in the original (each list holds EN + ES
phrases), so a single engine serves both routes; only the emitted referral wording and
the pregnancy/parental-leave disclaimer differ by language.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def _normalize(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.upper()
    s = s.replace("‘", "'").replace("’", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^A-Z0-9'\s-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return " " + s + " "


def _strip_channel_labels(t: str) -> str:
    for lbl in ("TAVERN_RAG", "NON_REPRESENTED_RAG", "REPRESENTED_RAG",
                "ENGLISH_EXTENDED_POLICY_TOOL", "ENGLISH_COMMON_POLICY_EXTENDED_TOOL"):
        t = re.sub(r"\b" + lbl + r"\b", " ", t)
    return t


def _phrase_re(p: str) -> "re.Pattern[str]":
    norm = _normalize(p).strip()
    return re.compile(r"\b" + re.escape(norm) + r"\b")


def _find_matches(t: str, phrases: List[str]) -> List[str]:
    return [p for p in phrases if _phrase_re(p).search(t)]


def _mask_phrases(t: str, phrases: List[str]) -> str:
    out = t
    for p in phrases:
        norm = _normalize(p).strip()
        if norm:
            out = re.sub(r"\b" + re.escape(norm) + r"\b", " ", out)
    return out


# ---------------------------------------------------------------------------
# HR / manager mention detection (reads the agent answer)
# ---------------------------------------------------------------------------
_PROPERTY_HR_PATTERNS = [
    r"\bPROPERTY('S)?\s+(HUMAN RESOURCES|HR)\b",
    r"\b(HUMAN RESOURCES|HR)\s+(DEPARTMENT|TEAM|OFFICE)?\s*(AT|OF|IN)\s+YOUR\s+PROPERTY\b",
    r"\bRECURSOS HUMANOS\s+DE\s+(SU|TU)\s+PROPIEDAD\b",
    r"\bDEPARTAMENTO DE RECURSOS HUMANOS\s+DE\s+(SU|TU)\s+PROPIEDAD\b",
]
_GENERIC_HR_PATTERNS = [r"\bHUMAN RESOURCES\b", r"\bHR DEPARTMENT\b", r"\bRECURSOS HUMANOS\b"]

_DIRECTED_MANAGER_PATTERNS = [
    r"\b(CONTACT|CONSULT|ASK|NOTIFY|INFORM|SEE|APPROACH|SPEAK|TALK|CHECK|REACH|COORDINATE|DISCUSS|SUBMIT|SUBMITTED)\s+(WITH\s+|TO\s+|OUT\s+TO\s+)?(YOUR|THEIR|THE|A)\s+((DIRECT|IMMEDIATE|DEPARTMENT|SHIFT|GENERAL|EXECUTIVE)\s+)?(SUPERVISOR|MANAGER|BOSS|LEADER|LEAD|DEPARTMENT HEAD)\b",
    r"\b(REFER|REFERRED|DIRECTED)\s+TO\s+(YOUR|THEIR|THE)\s+((DIRECT|IMMEDIATE|DEPARTMENT|SHIFT|GENERAL)\s+)?(SUPERVISOR|MANAGER|BOSS|LEADER|LEAD)\b",
    r"\b(COMUNIQUESE|HABLE|CONSULTE|PREGUNTE|CONTACTE|DIRIJASE|NOTIFIQUE|INFORME)\s+(CON|A|AL)\s+(SU|TU)\s+(SUPERVISOR|SUPERVISORA|GERENTE|JEFE|JEFA|ENCARGADO|ENCARGADA|LIDER)\b",
]
_GENERIC_MANAGER_PATTERNS = [
    r"\b(SUPERVISOR|SUPERVISORS|SUPERVISOR'S|MANAGER|MANAGERS|MANAGER'S|MANAGEMENT|BOSS|BOSSES|BOSS'S)\b",
    r"\b(DEPARTMENT|SHIFT|TEAM)\s+(HEAD|LEAD|LEADER)\b",
    r"\b(MANAGER|LEADER)\s+ON\s+DUTY\b",
    r"\b(SUPERVISORA|SUPERVISORES|GERENTE|GERENTES|GERENCIA|JEFE|JEFA|JEFES|ENCARGADO|ENCARGADA|LIDER)\b",
]

# HR_MENTION_STRICT / MANAGER_MENTION_STRICT are both False in v3.7/v3.8.
_HR_MENTION_PATTERNS = _PROPERTY_HR_PATTERNS + _GENERIC_HR_PATTERNS
_MANAGER_MENTION_PATTERNS = _DIRECTED_MANAGER_PATTERNS + _GENERIC_MANAGER_PATTERNS


def _as_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        for key in ("value", "output", "text"):
            if isinstance(v.get(key), str):
                return v[key]
        import json
        try:
            return json.dumps(v)
        except (TypeError, ValueError):
            return ""
    return str(v)


# ---------------------------------------------------------------------------
# First-person gate + anchor phrase tables (bilingual, verbatim from source)
# ---------------------------------------------------------------------------
_FIRST_PERSON_PHRASES = [
    "am i", "can i", "do i", "did i", "will i", "should i",
    "i can", "i take", "i get", "i have", "i had", "i'm", "i've",
    "i'll", "i'd", "i am", "i was", "i need", "i want", "help me",
    "my", "mine", "me", "i", "we", "our", "us",
    "yo", "mi", "mis", "mio", "mia", "mios", "mias",
    "conmigo", "nosotros", "nosotras", "nuestro", "nuestra",
    "nuestros", "nuestras",
    "puedo", "tengo", "necesito", "quiero", "voy", "estoy", "soy",
    "hice", "hago", "debo", "podria", "tendria", "querria",
    "tenemos", "podemos", "necesitamos", "queremos",
]

_ABSENCE_PHRASES = [
    "family medical leave", "family and medical leave", "fmla",
    "maternity leave", "maternity", "pregnancy leave",
    "pregnancy disability", "pregnant", "pregnancy",
    "expecting a baby", "expecting", "expectant",
    "americans with disabilities act", "ada leave", "ada",
    "reasonable accommodation", "workplace accommodation",
    "accommodation request", "interactive process",
    "undue hardship", "disability accommodation",
    "accommodation", "disability leave",
    "military leave", "active duty", "deployed", "deployment",
    "reservist", "reserve duty", "national guard", "guard duty", "userra",
    "domestic violence leave", "domestic violence",
    "domestic abuse", "protective order", "restraining order",
    "licencia familiar y medica", "licencia medica familiar",
    "licencia de maternidad", "permiso de maternidad",
    "maternidad", "embarazada", "embarazo",
    "licencia por embarazo", "esperando un bebe", "espero un bebe",
    "ley de estadounidenses con discapacidades",
    "adaptacion razonable", "acomodacion razonable",
    "solicitud de adaptacion", "adaptacion en el trabajo",
    "proceso interactivo", "carga excesiva", "licencia por discapacidad",
    "servicio militar", "licencia militar", "permiso militar",
    "reservista", "guardia nacional", "servicio activo",
    "violencia domestica", "licencia por violencia domestica",
    "abuso domestico", "orden de proteccion", "orden de restriccion",
]

_BENEFITS_PHRASES = [
    "health insurance", "medical insurance", "medical coverage",
    "medical plan", "medical", "prescription coverage",
    "dental care", "dental plan", "dental coverage", "dental",
    "dentist", "vision care", "vision plan", "vision coverage",
    "eye care", "eye coverage", "vision", "optometrist",
    "hsa", "health savings account",
    "fsa", "flexible spending account",
    "critical illness", "accident coverage", "accident insurance",
    "supplemental coverage", "supplemental insurance",
    "disability insurance", "disability benefits",
    "disability coverage", "short term disability",
    "long term disability", "disability",
    "life insurance", "hospital indemnity",
    "benefits enrollment", "open enrollment", "benefits",
    "insurance", "hospital coverage",
    "seguro medico", "seguro de salud", "cobertura medica",
    "plan medico", "cobertura de recetas", "receta medica",
    "recetas", "seguro dental", "plan dental", "cobertura dental",
    "dentista", "seguro de vision", "plan de vision",
    "cobertura de vision", "vision", "vista", "oculista",
    "optometrista", "cuenta de ahorros para la salud",
    "cuenta de gastos flexibles", "enfermedad critica",
    "cobertura de accidentes", "seguro de accidentes",
    "cobertura suplementaria", "seguro suplementario",
    "seguro por discapacidad", "beneficios por discapacidad",
    "discapacidad", "seguro de vida",
    "indemnizacion hospitalaria", "indemnizacion de hospital",
    "inscripcion de beneficios", "inscripcion abierta",
    "periodo de inscripcion", "beneficios", "seguro", "medico", "cobertura hospitalaria",
]

_HCM_PHRASES = [
    "policy document location", "personal details",
    "personal information", "employment details",
    "employment information", "job information", "current job",
    "current jobs", "job title", "pay rate", "salary",
    "citizenship", "email address", "work email", "email",
    "detalles personales", "informacion personal",
    "detalles de empleo", "informacion de empleo",
    "informacion del trabajo", "trabajo actual", "puesto actual",
    "titulo del puesto", "tarifa de pago", "tasa salarial",
    "tasa de pago", "salario", "sueldo", "ciudadania",
    "correo electronico", "correo del trabajo", "correo laboral",
    "ubicacion del documento de politica",
]

_ADP_PHRASES = [
    "paycheck amount", "pay history", "pay stub", "paystub",
    "pto balance", "time card", "timecard", "punch in",
    "punch out", "clock in", "clock out", "pay date", "paydate", "tax", "tax forms",
    "tax statement", "tax statements", "tax withholding", "tax documents", "tax details",
    "withholding", "tax form", "w2", "w-2", "direct deposit",
    "paycheck", "pay check", "payroll", "pto", "pay rate", "pay",
    "vacation", "sick leave", "sick day", "sick days",
    "monto del cheque", "historial de pago", "talon de pago",
    "saldo de pto", "tarjeta de tiempo", "reloj de entrada",
    "reloj de salida", "marcar entrada", "marcar salida",
    "fecha de pago", "dia de pago", "declaracion de impuestos",
    "retencion de impuestos", "impuestos", "formulario w2",
    "deposito directo", "cheque de pago", "nomina",
    "vacaciones", "licencia por enfermedad", "dia de enfermedad",
    "dias de enfermedad", "pago", "impuesto", "formularios de impuestos",
    "documentos fiscales", "detalles de impuestos",
]

_MANAGER_PHRASES = [
    "shift schedule", "schedule shift", "work schedule", "my schedule",
    "next shift", "my shift", "shift", "shifts",
    "schedule", "roster", "rota",
    "when do i work", "when am i working", "who is working",
    "scheduling", "scheduled",
    "horario de turno", "horario de trabajo", "mi horario",
    "turno de trabajo", "proximo turno", "mi turno",
    "turno", "turnos", "horario", "cuando trabajo",
]

_HR_PHRASES = [
    "fired", "get fired", "getting fired", "terminated",
    "termination", "terminate", "let go", "laid off", "layoff",
    "lay off", "dismissal", "dismissed", "wrongful termination",
    "disciplinary action", "disciplinary", "discipline",
    "verbal warning", "written warning", "final warning",
    "write up", "written up", "write-up", "suspension", "suspended",
    "attendance points", "misconduct", "insubordination",
    "appeal", "team member council", "grievance",
    "bereavement leave", "bereavement",
    "passed away", "pass away", "passing away", "passing of",
    "death in the family", "death in my family",
    "family member died", "family member passed",
    "loss of a loved one", "loss of my", "lost my",
    "died", "death", "deceased", "funeral", "attend a funeral",
    "paid parental leave", "parental leave", "ppl",
    "paternity leave", "paternity", "bonding leave",
    "child bonding", "adoption leave", "adopting a child",
    "adopt a child", "adopted a child", "adoption",
    "foster leave", "fostering a child", "foster a child",
    "foster child", "birth of a child", "newborn",
    "having a baby", "paid holiday", "paid holidays", "holiday", "holidays", "holiday pay",
    "voting leave", "time off to vote", "leave to vote",
    "election day", "go vote", "voting",
    "school activities leave", "school activity",
    "school activities", "school event", "school events",
    "parent teacher conference", "parent-teacher conference",
    "school exigent", "school emergency", "school emergencies",
    "child's school",
    "jury duty", "jury service", "jury summons", "witness duty",
    "court summons", "subpoena", "subpoenaed", "jury",
    "blackout days", "blackout day", "blackout dates",
    "blackout date", "peak business days", "peak business day", "confidential",
    "licencia por duelo", "licencia por fallecimiento",
    "permiso por duelo", "permiso por fallecimiento",
    "fallecio", "fallecimiento", "murio", "muerte",
    "difunto", "difunta", "duelo", "luto",
    "muerte en la familia", "perdi a mi", "perdi a un",
    "asistir a un funeral",
    "licencia parental pagada", "licencia parental",
    "permiso parental", "licencia de paternidad",
    "permiso de paternidad", "paternidad",
    "licencia por adopcion", "permiso por adopcion",
    "adopcion", "adoptar", "adoptando", "cuidado de crianza",
    "vinculacion con el bebe", "nacimiento", "recien nacido", "bebe",
    "permiso para votar", "licencia para votar",
    "dia de elecciones", "dia electoral", "votar",
    "actividades escolares", "actividad escolar",
    "evento escolar", "eventos escolares",
    "emergencia escolar", "emergencias escolares",
    "reunion de padres", "escuela de mi hijo",
    "servicio de jurado", "deber de jurado", "jurado",
    "citacion judicial", "citacion", "testigo",
    "dias de apagon", "dia de apagon", "dias pico", "confidencial",
    "dia festivo", "dias festivos", "dia festivo pagado",
    "dias festivos pagados", "feriado", "feriados",
    "feriado pagado", "feriados pagados",
]

_REPORTING_MANAGER_ISSUE = [
    re.compile(r"\b(REPORT|REPORTING|COMPLAIN|COMPLAINT|ISSUE|PROBLEM|CONCERN|HARASSMENT|GRIEVANCE)\b.*\b(SUPERVISOR|MANAGER|BOSS)\b"),
    re.compile(r"\b(SUPERVISOR|MANAGER|BOSS)\b.*\b(REPORT|REPORTING|COMPLAIN|COMPLAINT|ISSUE|PROBLEM|CONCERN|HARASSMENT|GRIEVANCE)\b"),
]

# Pregnancy / parental-leave disclaimer trigger.
_DISCLAIMER_PHRASES = [
    "pregnancy leave", "maternity leave", "paternity leave", "parental leave",
    "paid parental leave", "pregnancy accommodation", "pregnancy accommodations",
    "pregnant workers fairness", "pregnant workers fairness act", "expecting a baby",
    "having a baby", "baby due", "due date", "maternity benefits",
    "licencia por embarazo", "licencia de maternidad", "licencia de paternidad",
    "licencia parental", "licencia parental pagada", "esperando un bebe",
    "espero un bebe", "esperando un hijo", "espero un hijo", "tener un bebe",
    "tener un hijo", "recien nacido", "bebe", "hijo",
]
_DISCLAIMER_STEMS = ["pregn", "matern", "patern", "parental", "bond", "baby",
                     "newborn", "bebe", "hijo"]
_DISCLAIMER_EXACT = ["pwfa"]

_PORTAL_SYSTEMS = ["ADP", "Benefits Connect", "Absence Resources", "HCM"]
_ALLOWED = ["ADP", "Benefits Connect", "Absence Resources", "Ask your Manager", "HCM", "Human Resources"]


# ---------------------------------------------------------------------------
# Language-specific message catalog
# ---------------------------------------------------------------------------
MESSAGES = {
    "EN": {
        "conjunction": " and ",
        "manager_suffix": " Please reach out to your Manager for further details.",
        "manager_only": "Please reach out to your Manager for further details.",
        "manager_and_hr": "Please reach out to your Manager for further details, and for more information, please contact your property Human Resources.",
        "hr_property": "For more information, please contact your property Human Resources.",
        "only_human": "For more details, please contact your property {joined}.",
        "hcm": {
            True: "Please visit Oracle HCM to review your personal details, your employment details (including base salary), and the full policy documents, including the Team Member Handbook.",
            False: "Please visit Oracle HCM to review your personal details, your employment details (including base salary), and the full policy documents, including the Team Member Handbook. For further questions, please contact Human Resources.",
        },
        "adp": {
            True: "You may find helpful information in ADP, where you can find your pay stub, time card, tax withholding information, direct deposit information, and annual tax statement (W-2) / annual tax form (W-2). ",
            False: "You may find helpful information in ADP, where you can find your pay stub, time card, tax withholding information, direct deposit information, and annual tax statement (W-2) / annual tax form (W-2). If you still have questions, please contact Human Resources.",
        },
        "benefits": {
            True: "For more information about your benefits such as medical, dental, vision, disability insurance, life insurance, or voluntary benefit products, visit Benefit Connect. You can also enroll in benefits, apply to make changes for a Qualifying Life Event (QLE), or review your current coverage on Benefit Connect.",
            False: "For more information about your benefits such as medical, dental, vision, disability insurance, life insurance, or voluntary benefit products, visit Benefit Connect. You can also enroll in benefits, apply to make changes for a Qualifying Life Event (QLE), or review your current coverage on Benefit Connect. For further questions or assistance, contact Human Resources.",
        },
        "absence": {
            True: "To review your eligibility for protected leave such as FMLA, ADA, military leave, or other leaves protected by state or federal statute, please visit Absence Resources.",
            False: "To review your eligibility for protected leave such as FMLA, ADA, military leave, or other leaves protected by state or federal statute, please visit Absence Resources. For further questions or assistance, contact Human Resources.",
        },
        "disclaimer_marker": "there are a number of leave resources",
        "disclaimer": {
            "TAVERN": {
                True: "There are a number of leave resources or workplace accommodations that may be available through FMLA, ADA, Nevada Pregnant Fairness Workers’ Act, personal leave. Each leave type has different eligibility requirements. To better understand leave that is available to you, refer Absence Resources.",
                False: "There are a number of leave resources or workplace accommodations that may be available through FMLA, ADA, Nevada Pregnant Fairness Workers’ Act, personal leave. Each leave type has different eligibility requirements. To better understand leave that is available to you, refer Absence Resources or reach out to your Human Resources department.",
            },
            "OTHER": {
                True: "There are a number of leave resources or workplace accommodations that may be available through FMLA, ADA, Nevada Pregnant Fairness Workers’ Act, personal leave, or Paid Parental Leave. Each leave type has different eligibility requirements. To better understand leave that is available to you, refer Absence Resources.",
                False: "There are a number of leave resources or workplace accommodations that may be available through FMLA, ADA, Nevada Pregnant Fairness Workers’ Act, personal leave, or Paid Parental Leave. Each leave type has different eligibility requirements. To better understand leave that is available to you, refer Absence Resources or reach out to your Human Resources department.",
            },
        },
    },
    "ES": {
        "conjunction": " y ",
        "manager_suffix": " Comuníquese con su gerente para obtener más información.",
        "manager_only": "Comuníquese con su gerente para obtener más información.",
        "manager_and_hr": "Comuníquese con su gerente para obtener más información y, para más información, comuníquese con el departamento de Recursos Humanos de su propiedad.",
        "hr_property": "Para obtener más información, comuníquese con el departamento de Recursos Humanos de su propiedad.",
        "only_human": "Para obtener más información, comuníquese con el departamento de Recursos Humanos de su propiedad.",
        "hcm": {
            True: "Visite Oracle HCM para revisar sus datos personales, los detalles de su empleo (incluido su salario base) y los documentos completos de políticas, incluido el Manual para Miembros del Equipo.",
            False: "Visite Oracle HCM para revisar sus datos personales, los detalles de su empleo (incluido su salario base) y los documentos completos de políticas, incluido el Manual para Miembros del Equipo. Si tiene más preguntas, comuníquese con Recursos Humanos.",
        },
        "adp": {
            True: "Puede encontrar información útil en ADP, donde podrá consultar su recibo de pago, tarjeta de tiempo, información sobre retenciones de impuestos, información de depósito directo y su declaración anual de impuestos (W-2).",
            False: "Puede encontrar información útil en ADP, donde podrá consultar su recibo de pago, tarjeta de tiempo, información sobre retenciones de impuestos, información de depósito directo y su declaración anual de impuestos (W-2). Si aún tiene preguntas, comuníquese con Recursos Humanos.",
        },
        "benefits": {
            True: "Para obtener más información sobre sus beneficios, como seguro médico, dental, de visión, seguro por discapacidad, seguro de vida o productos de beneficios voluntarios, visite Benefits Connect. También puede inscribirse en los beneficios, solicitar cambios debido a un Evento de Vida Calificado (QLE) o revisar su cobertura actual en Benefits Connect.",
            False: "Para obtener más información sobre sus beneficios, como seguro médico, dental, de visión, seguro por discapacidad, seguro de vida o productos de beneficios voluntarios, visite Benefits Connect. También puede inscribirse en los beneficios, solicitar cambios debido a un Evento de Vida Calificado (QLE) o revisar su cobertura actual en Benefits Connect. Si necesita más información o asistencia, comuníquese con Recursos Humanos.",
        },
        "absence": {
            True: "Para revisar su elegibilidad para licencias protegidas, como FMLA, ADA, licencia militar u otras licencias protegidas por leyes estatales o federales, visite Absence Resources.",
            False: "Para revisar su elegibilidad para licencias protegidas, como FMLA, ADA, licencia militar u otras licencias protegidas por leyes estatales o federales, visite Absence Resources. Si necesita más información o asistencia, comuníquese con Recursos Humanos.",
        },
        "disclaimer_marker": "existen varios recursos de licencia",
        "disclaimer": {
            "TAVERN": {
                True: "Existen varios recursos de licencia o adaptaciones en el lugar de trabajo que pueden estar disponibles a través de la FMLA, la ADA, la Ley de Equidad para Trabajadoras Embarazadas de Nevada y la licencia personal. Cada tipo de licencia tiene diferentes requisitos de elegibilidad. Para comprender mejor qué licencia puede estar disponible para usted, consulte Absence Resources.",
                False: "Existen varios recursos de licencia o adaptaciones en el lugar de trabajo que pueden estar disponibles a través de la FMLA, la ADA, la Ley de Equidad para Trabajadoras Embarazadas de Nevada y la licencia personal. Cada tipo de licencia tiene diferentes requisitos de elegibilidad. Para comprender mejor qué licencia puede estar disponible para usted, consulte Absence Resources o comuníquese con su departamento de Recursos Humanos.",
            },
            "OTHER": {
                True: "Existen varios recursos de licencia o adaptaciones en el lugar de trabajo que pueden estar disponibles a través de la FMLA, la ADA, la Ley de Equidad para Trabajadoras Embarazadas de Nevada, la licencia personal o la Licencia Parental Pagada. Cada tipo de licencia tiene diferentes requisitos de elegibilidad. Para comprender mejor qué licencia puede estar disponible para usted, consulte Absence Resources.",
                False: "Existen varios recursos de licencia o adaptaciones en el lugar de trabajo que pueden estar disponibles a través de la FMLA, la ADA, la Ley de Equidad para Trabajadoras Embarazadas de Nevada, la licencia personal o la Licencia Parental Pagada. Cada tipo de licencia tiene diferentes requisitos de elegibilidad. Para comprender mejor qué licencia puede estar disponible para usted, consulte Absence Resources o comuníquese con su departamento de Recursos Humanos.",
            },
        },
    },
}


def _join_systems(list_names: List[str], conj: str) -> str:
    if not list_names:
        return ""
    if len(list_names) == 1:
        return list_names[0]
    if len(list_names) == 2:
        return list_names[0] + conj + list_names[1]
    return ", ".join(list_names[:-1]) + conj + list_names[-1]


def fetch_system_mapping(agent_answer: Any, raw_question: str, tm_type: str,
                         query_intent: str, rag_value: Any, language: str = "EN") -> Dict[str, Any]:
    """Port of FETCH_SYSTEM_MAPPING_SCRIPT.

    Returns a dict with ``question_type``, ``system_mapping``, ``referral_message``,
    ``answer_mentions_property_hr`` and ``answer_mentions_manager``.
    """
    lang = "ES" if str(language).upper() == "ES" else "EN"
    msg = MESSAGES[lang]
    tm_type = (tm_type or "").upper()

    policy_raw = str("" if rag_value is None else rag_value)[:20000]
    question_text = _normalize(raw_question)
    policy_text = _strip_channel_labels(_normalize(policy_raw))

    answer_text = _normalize(_as_text(agent_answer)[:20000])
    answer_mentions_property_hr = (
        answer_text.strip() != "" and any(re.search(p, answer_text) for p in _HR_MENTION_PATTERNS)
    )
    answer_mentions_manager = (
        answer_text.strip() != "" and any(re.search(p, answer_text) for p in _MANAGER_MENTION_PATTERNS)
    )

    is_personal = any(_phrase_re(p).search(question_text) for p in _FIRST_PERSON_PHRASES)

    systems: List[str] = []
    if is_personal:
        absence_hits = _find_matches(question_text, _ABSENCE_PHRASES)
        benefits_hits = _find_matches(question_text, _BENEFITS_PHRASES)
        hcm_hits = _find_matches(question_text, _HCM_PHRASES)

        reporting_manager_issue = any(p.search(question_text) for p in _REPORTING_MANAGER_ISSUE)
        manager_hits = [] if reporting_manager_issue else _find_matches(question_text, _MANAGER_PHRASES)
        hr_hits = ["Human Resources"] if reporting_manager_issue else _find_matches(question_text, _HR_PHRASES)

        adp_text = _mask_phrases(question_text, absence_hits + benefits_hits + hcm_hits + manager_hits + hr_hits)
        adp_hits = _find_matches(adp_text, _ADP_PHRASES)

        if absence_hits:
            systems = ["Absence Resources"]
        elif benefits_hits:
            systems = ["Benefits Connect"]
        elif adp_hits:
            systems = ["ADP"]
        elif manager_hits:
            systems = ["Ask your Manager"]
        elif hcm_hits:
            systems = ["HCM"]
        elif hr_hits:
            systems = ["Human Resources"]

        # Pass 2: policy-context rescue for ambiguous questions.
        if not systems and policy_text.strip():
            def prep(phrases: List[str]) -> List[str]:
                seen, out = set(), []
                for p in phrases:
                    n = _normalize(p).strip()
                    if n and n not in seen:
                        seen.add(n)
                        out.append(n)
                return sorted(out, key=len, reverse=True)

            def score_anchor(t: str, phrases: List[str]) -> int:
                work, n = t, 0
                for norm in prep(phrases):
                    rx = re.compile(r"\b" + re.escape(norm) + r"\b")
                    matches = rx.findall(work)
                    if matches:
                        n += len(matches)
                        work = rx.sub(" ", work)
                return n

            winners = (_find_matches(policy_text, _ABSENCE_PHRASES)
                       + _find_matches(policy_text, _BENEFITS_PHRASES)
                       + _find_matches(policy_text, _HCM_PHRASES)
                       + _find_matches(policy_text, _MANAGER_PHRASES)
                       + _find_matches(policy_text, _HR_PHRASES))
            adp_ctx = score_anchor(_mask_phrases(policy_text, winners), _ADP_PHRASES)
            scores = sorted([
                ("Absence Resources", score_anchor(policy_text, _ABSENCE_PHRASES)),
                ("Benefits Connect", score_anchor(policy_text, _BENEFITS_PHRASES)),
                ("ADP", adp_ctx),
                ("Ask your Manager", score_anchor(policy_text, _MANAGER_PHRASES)),
                ("HCM", score_anchor(policy_text, _HCM_PHRASES)),
                ("Human Resources", score_anchor(policy_text, _HR_PHRASES)),
            ], key=lambda x: x[1], reverse=True)
            if scores[0][1] >= 2 and scores[0][1] >= 2 * scores[1][1]:
                systems = [scores[0][0]]

        if not systems:
            systems = ["Human Resources"]

    # --- Referral message generation ---
    valid_systems = [s for s in (systems if is_personal else []) if s in _ALLOWED]
    only_human = bool(valid_systems) and all(s == "Human Resources" for s in valid_systems)

    suppressed: List[str] = []
    if answer_mentions_property_hr:
        suppressed.append("Human Resources")
    if answer_mentions_manager:
        suppressed.append("Ask your Manager")
    referral_systems = [s for s in valid_systems if s not in suppressed]

    display_names = list(referral_systems)
    has_manager = "Ask your Manager" in display_names
    list_names = [s for s in display_names if s != "Ask your Manager"]
    joined = _join_systems(list_names, msg["conjunction"])
    manager_suffix = msg["manager_suffix"] if has_manager else ""

    if joined == "" and not has_manager:
        referral_message = msg["hr_property"] if (valid_systems and not answer_mentions_property_hr) else ""
    elif referral_systems == ["HCM"]:
        referral_message = msg["hcm"][answer_mentions_property_hr] + manager_suffix
    elif referral_systems == ["ADP"]:
        referral_message = msg["adp"][answer_mentions_property_hr] + manager_suffix
    elif referral_systems == ["Benefits Connect"]:
        referral_message = msg["benefits"][answer_mentions_property_hr] + manager_suffix
    elif referral_systems == ["Absence Resources"]:
        referral_message = msg["absence"][answer_mentions_property_hr] + manager_suffix
    elif answer_mentions_property_hr:
        referral_message = msg["manager_only"] if has_manager else ""
    elif only_human and lang == "EN":
        referral_message = msg["only_human"].format(joined=joined)
    elif has_manager:
        referral_message = msg["manager_and_hr"]
    else:
        referral_message = msg["hr_property"]

    # --- Pregnancy / parental-leave disclaimer injection ---
    text = question_text.lower()
    words = re.findall(r"[a-z0-9]+", text)
    needs_disclaimer = (
        any(_phrase_re(p).search(question_text) for p in _DISCLAIMER_PHRASES)
        or any(any(w.startswith(stem) for w in words) for stem in _DISCLAIMER_STEMS)
        or any(w in words for w in _DISCLAIMER_EXACT)
    )
    if needs_disclaimer and msg["disclaimer_marker"] not in referral_message.lower():
        bucket = "TAVERN" if tm_type == "TAVERN" else "OTHER"
        referral_message = msg["disclaimer"][bucket][answer_mentions_property_hr]

    output_systems = ["NONE"]
    if is_personal:
        output_systems = [s for s in systems if s != "Ask your Manager"] if answer_mentions_manager else systems
        if not output_systems:
            output_systems = ["Human Resources"]

    return {
        "question_type": "PERSONAL" if is_personal else "POLICY",
        "system_mapping": output_systems,
        "referral_message": referral_message,
        "answer_mentions_property_hr": answer_mentions_property_hr,
        "answer_mentions_manager": answer_mentions_manager,
    }
