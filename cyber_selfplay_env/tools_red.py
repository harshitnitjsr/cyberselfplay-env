RED_TOOLS = {
    "recon_network",
    "enumerate_services",
    "attempt_exploit",
    "dump_credentials",
    "pivot_host",
    "establish_persistence",
    "prepare_exfiltration",
    "execute_exfiltration",
    "cover_tracks",
    "sabotage_recovery_plan",
}


def validate_red_tool(tool_name: str) -> bool:
    return tool_name in RED_TOOLS
