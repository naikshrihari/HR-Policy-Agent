"""Python ports of the Oracle workflow's JavaScript ``CODE`` nodes.

Each function mirrors one ``sourceCode`` block from the original data pipeline.  The
JS convention of returning a bare ``result`` object is preserved: every function
returns the value that the node placed on ``$output``.
"""
