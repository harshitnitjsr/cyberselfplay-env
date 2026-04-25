BLUE_TOOLS = {
    "query_siem",
    "triage_alerts",
    "isolate_host",
    "disable_account",
    "rotate_secrets",
    "deploy_patch",
    "harden_policy",
    "restore_backup",
    "run_forensics",
    "publish_ioc_blocklist",
    "execute_instruction",
    "checkpoint_plan",
    "reconcile_state",
}


def validate_blue_tool(tool_name: str) -> bool:
    return tool_name in BLUE_TOOLS
