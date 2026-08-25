"""External service clients (HCM REST, user session, chat store, RAG document tools).

Each client has a live implementation (Oracle Fusion / your own endpoints) and a mock
default so the whole graph runs offline.  Which one is used is chosen from
:class:`hr_policy_agent.config.Settings`.
"""
