from __future__ import annotations

import json
from typing import Any, Callable

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound, ResourceDoesNotExist

from scripts.common import SETUP_STATE_PATH, StateStore
from scripts.setup_df1 import EXPECTED_USER, PROFILE


def _delete(
    state: dict[str, Any],
    store: StateStore,
    kind: str,
    name: str,
    action: Callable[[], Any],
) -> None:
    deleted = state.setdefault("deleted", [])
    marker = {"kind": kind, "name": name}
    if marker in deleted:
        return
    try:
        action()
    except (NotFound, ResourceDoesNotExist):
        pass
    deleted.append(marker)
    store.save(state)


def teardown() -> dict[str, Any]:
    store = StateStore(SETUP_STATE_PATH)
    state = store.load()
    client = WorkspaceClient(profile=PROFILE)
    current = client.current_user.me()
    if current.user_name != EXPECTED_USER:
        raise RuntimeError(f"df1 caller mismatch: {current.user_name}")
    if state.get("workspace_host") != client.config.host:
        raise RuntimeError("setup state belongs to another workspace")

    deployment = state.get("deployment") or {}
    if deployment.get("app_name"):
        _delete(
            state,
            store,
            "app",
            deployment["app_name"],
            lambda: client.apps.delete(deployment["app_name"]),
        )
    if state.get("space_id"):
        _delete(
            state,
            store,
            "genie",
            state["space_id"],
            lambda: client.genie.delete_space(state["space_id"]),
        )
    _delete(
        state,
        store,
        "schema",
        state["schema"],
        lambda: client.schemas.delete(state["schema"], force=True),
    )
    state["teardown_complete"] = True
    store.save(state)
    return state


def main() -> None:
    print(json.dumps(teardown().get("deleted", []), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
