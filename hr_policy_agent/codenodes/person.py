"""Ports of ENCRYPT_PERSON_NUMBER and RETRIEVE_PERSON_DETAILS_SCRIPT."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional


def encrypt_person_number(person_number: Optional[str]) -> Dict[str, str]:
    """Port of the ENCRYPT_PERSON_NUMBER code node.

    The original re-implemented SHA-256 in JS; here we use hashlib.  Behaviour is
    identical: a null/undefined person number falls back to "123456789"; any value
    whose length is not exactly 15 is hashed and the uppercased hex is truncated to
    15 characters, otherwise the value is passed through unchanged.
    """
    if person_number is None:
        person_number = "123456789"
    pn = str(person_number)
    if len(pn) != 15:
        digest = hashlib.sha256(pn.encode("utf-8")).hexdigest().upper()
        person_number_hash = digest[:15]
    else:
        person_number_hash = pn
    return {"person_number_hash": person_number_hash}


def _is_bu_code_meaningful(bu_code: Any) -> bool:
    if not bu_code or not isinstance(bu_code, str):
        return False
    stripped = bu_code.strip()
    if stripped == "":
        return False
    return stripped.upper() not in {"N/A", "NA", "NONE"}


def retrieve_person_details(hcm_data: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Port of RETRIEVE_PERSON_DETAILS_SCRIPT.

    Traverses the Oracle HCM ``workers`` response (items -> workRelationships ->
    assignments), selects the single ACTIVE + PRIMARY + type E/C assignment, and
    classifies the Team Member type:

      * LegalEmployerName contains "SCT"     -> "Tavern"
      * meaningful BargainingUnitCode        -> "Represented"
      * otherwise                            -> "Non Represented"
    """
    result = {
        "tmDepartment": "",
        "tmProperty": "",
        "locationName": "",
        "tmType": "Non Represented",
    }

    items = (hcm_data or {}).get("items")
    if not items or not isinstance(items, list):
        return result

    active_assignment: Optional[Dict[str, Any]] = None
    active_legal_employer = ""

    for person in items:
        work_rels = person.get("workRelationships") if isinstance(person, dict) else None
        if not isinstance(work_rels, list):
            continue
        for wr in work_rels:
            assignments = wr.get("assignments") if isinstance(wr, dict) else None
            if not isinstance(assignments, list):
                continue
            for assign in assignments:
                is_active = assign.get("AssignmentStatusType") == "ACTIVE"
                is_primary = assign.get("PrimaryFlag") in (True, "TRUE")
                is_type_valid = assign.get("AssignmentType") in ("E", "C")
                if is_active and is_primary and is_type_valid:
                    active_assignment = assign
                    active_legal_employer = wr.get("LegalEmployerName") or ""
                    break
            if active_assignment:
                break
        if active_assignment:
            break

    if not active_assignment:
        return result

    determined_tm_type = "Non Represented"
    if "SCT" in active_legal_employer:
        determined_tm_type = "Tavern"
    elif _is_bu_code_meaningful(active_assignment.get("BargainingUnitCode")):
        determined_tm_type = "Represented"

    result["tmDepartment"] = active_assignment.get("DepartmentName") or ""
    result["tmProperty"] = active_legal_employer or ""
    result["locationName"] = active_assignment.get("LocationName") or ""
    result["tmType"] = determined_tm_type
    return result
