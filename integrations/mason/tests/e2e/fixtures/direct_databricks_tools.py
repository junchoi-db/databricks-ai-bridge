"""Directly authored Databricks integrations for the Mason E2E matrix."""

import databricks_mason.integrations as mason_integrations

DATABRICKS_TOOLS: tuple[mason_integrations.Integration, ...] = (
    mason_integrations.Sandbox(
        id="sandbox",
        scopes=(
            mason_integrations.Scope.table(
                "__SANDBOX_TABLE__",
                permission="read_only",
            ),
        ),
        auth="app",
    ),
    mason_integrations.MCPService(
        id="google_drive",
        service="system.ai.google_drive",
        auth="user",
    ),
    mason_integrations.UCFunction(
        id="mason_uc_marker",
        function="__UC_FUNCTION__",
    ),
)
