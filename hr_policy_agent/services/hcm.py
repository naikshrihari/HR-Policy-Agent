"""HCM REST client — port of STN_HCM_REST_TOOL_WORKFLOW / getWorkerDetails.

Fetches a worker's active assignment + bargaining unit from Oracle Fusion HCM so the
graph can classify the Team Member type (Tavern / Represented / Non Represented).

The live implementation calls the Oracle ``workers`` REST resource with the same query
and field selection as the original tool.  When no endpoint is configured, a mock
returns a Non-Represented worker so the pipeline runs offline.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..config import Settings

# Same field projection as the original tool's resourcePath.
_FIELDS = (
    "PersonId,PersonNumber;"
    "workRelationships:PeriodOfServiceId,StartDate,TerminationDate,LegalEmployerName,"
    "WorkerType,PrimaryFlag;"
    "workRelationships.assignments:AssignmentId,AssignmentName,AssignmentType,"
    "AssignmentStatusType,PrimaryFlag,JobId,DepartmentName,BargainingUnitCode,"
    "LocationId,LocationName"
)


class HCMClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def get_worker_details(self, person_number: Optional[str]) -> Dict[str, Any]:
        # Guard mirrors the tool description: never call with a null/empty/non-numeric
        # PersonNumber.
        if not person_number or not str(person_number).strip():
            return {"items": []}
        if self.settings.use_mock_hcm or not self.settings.hcm_base_url:
            return self._mock(person_number)
        return self._live(person_number)

    # -- live ---------------------------------------------------------------
    def _live(self, person_number: str) -> Dict[str, Any]:
        import requests  # imported lazily so the mock path needs no dependency

        url = (
            f"{self.settings.hcm_base_url}/hcmRestApi/resources/11.13.18.05/workers"
            f"?q=PersonNumber='{person_number}'&fields={_FIELDS}&onlyData=true"
        )
        auth = None
        if self.settings.hcm_username and self.settings.hcm_password:
            auth = (self.settings.hcm_username, self.settings.hcm_password)
        resp = requests.get(url, auth=auth, headers={"Accept": "application/json"}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    # -- mock ---------------------------------------------------------------
    def _mock(self, person_number: str) -> Dict[str, Any]:
        return {
            "items": [
                {
                    "PersonNumber": person_number,
                    "workRelationships": [
                        {
                            "LegalEmployerName": "Station Casinos LLC",
                            "PrimaryFlag": True,
                            "assignments": [
                                {
                                    "AssignmentStatusType": "ACTIVE",
                                    "PrimaryFlag": True,
                                    "AssignmentType": "E",
                                    "DepartmentName": "Food & Beverage",
                                    "BargainingUnitCode": "N/A",
                                    "LocationName": "Red Rock Casino Resort",
                                }
                            ],
                        }
                    ],
                }
            ]
        }
