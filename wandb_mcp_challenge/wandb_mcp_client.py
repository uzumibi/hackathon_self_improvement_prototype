import json
import os
from typing import Any, Dict, List, Optional

import requests


class WandbMCPClient:
    """
    Minimal JSON-RPC client for the hosted W&B MCP endpoint.
    The goal is resilience: even if tool names differ by version, we keep the loop running.
    """

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.base_url = base_url or os.getenv("WANDB_MCP_URL", "https://mcp.withwandb.com/mcp")
        self.api_key = api_key or os.getenv("WANDB_API_KEY")
        self._initialized = False

    @staticmethod
    def _parse_sse_json(text: str) -> Dict[str, Any]:
        for line in text.splitlines():
            if line.startswith("data:"):
                raw = line[len("data:") :].strip()
                if not raw:
                    continue
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    continue
        raise ValueError("No JSON data frame found in SSE response")

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("WANDB_API_KEY is not set")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2025-03-26",
        }
        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

        content_type = (response.headers.get("content-type") or "").lower()
        if "text/event-stream" in content_type:
            return self._parse_sse_json(response.text)
        return response.json()

    def initialize(self) -> Dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": "initialize",
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {
                    "name": "wandb_mcp_challenge_client",
                    "version": "0.1.0",
                },
            },
        }
        res = self._post(payload)

        try:
            self._post(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {},
                }
            )
        except Exception:
            pass

        self._initialized = True
        return res

    def list_tools(self) -> Dict[str, Any]:
        if not self._initialized:
            self.initialize()
        payload = {
            "jsonrpc": "2.0",
            "id": "list-tools",
            "method": "tools/list",
            "params": {},
        }
        return self._post(payload)

    def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": f"call-{tool_name}",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments or {},
            },
        }
        return self._post(payload)

    @staticmethod
    def _extract_tool_names(tool_listing: Dict[str, Any]) -> List[str]:
        result = tool_listing.get("result", {})
        tools = result.get("tools", []) if isinstance(result, dict) else []
        names: List[str] = []
        for t in tools:
            if isinstance(t, dict):
                name = t.get("name")
                if isinstance(name, str):
                    names.append(name)
        return names

    @staticmethod
    def _find_metric_tree(obj: Any) -> Optional[Dict[str, Any]]:
        if isinstance(obj, dict):
            has_target_key = any(
                k in obj for k in ["overall_mae", "within_tolerance_rate", "mae_per_axis", "summary", "metrics"]
            )
            if has_target_key:
                return obj
            for v in obj.values():
                found = WandbMCPClient._find_metric_tree(v)
                if found is not None:
                    return found
        if isinstance(obj, list):
            for item in obj:
                found = WandbMCPClient._find_metric_tree(item)
                if found is not None:
                    return found
        return None

    def _call_tool_with_variants(self, tool_name: str, arg_variants: List[Dict[str, Any]]) -> Dict[str, Any]:
        attempted: List[Dict[str, Any]] = []
        for args in arg_variants:
            clean_args = {k: v for k, v in args.items() if v is not None}
            try:
                raw = self.call_tool(tool_name, clean_args)
                return {
                    "ok": True,
                    "tool_name": tool_name,
                    "arguments": clean_args,
                    "raw": raw,
                }
            except Exception as e:
                attempted.append(
                    {
                        "tool_name": tool_name,
                        "arguments": clean_args,
                        "error": str(e),
                    }
                )
        return {
            "ok": False,
            "tool_name": tool_name,
            "attempted": attempted,
        }

    def safe_fetch_latest_metrics(
        self,
        entity: Optional[str] = None,
        project: Optional[str] = None,
        loop_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Best-effort MCP inspection for self-improvement loop.
        The loop attempts all discovered W&B/Weave tools in this order:
        1) query_wandb_entity_projects
        2) query_wandb_tool
        3) count_weave_traces_tool
        4) query_weave_traces_tool
        5) query_wandb_support_bot
        6) create_wandb_report_tool
        """
        try:
            listed = self.list_tools()
            names = self._extract_tool_names(listed)
        except Exception as e:
            return {"mcp_ok": False, "error": f"tools/list failed: {e}"}

        required_tools = [
            "query_weave_traces_tool",
            "count_weave_traces_tool",
            "query_wandb_tool",
            "create_wandb_report_tool",
            "query_wandb_entity_projects",
            "query_wandb_support_bot",
        ]
        available_tools = [t for t in required_tools if t in names]

        if not available_tools:
            return {
                "mcp_ok": True,
                "tools_discovered": names,
                "run_query": None,
                "message": "No required W&B MCP loop tools discovered.",
            }

        lc = loop_context or {}
        iter_idx = lc.get("iteration")
        overall_mae = lc.get("overall_mae")
        worst_axis = lc.get("worst_axis")

        snapshot: Dict[str, Any] = {
            "mcp_ok": True,
            "tools_discovered": names,
            "loop_tool_policy": {
                "required_tools": required_tools,
                "available_tools": available_tools,
                "message": "Use all available W&B MCP loop tools every iteration for inspect/diagnose/report.",
            },
            "tool_results": {},
        }

        if "query_wandb_entity_projects" in names:
            snapshot["tool_results"]["query_wandb_entity_projects"] = self._call_tool_with_variants(
                "query_wandb_entity_projects",
                [
                    {"entity": entity},
                    {},
                ],
            )

        if "query_wandb_tool" in names:
            run_result = self._call_tool_with_variants(
                "query_wandb_tool",
                [
                    {"entity": entity, "project": project},
                    {"project": project},
                    {"entity": entity},
                    {},
                ],
            )
            snapshot["tool_results"]["query_wandb_tool"] = run_result
            if run_result.get("ok"):
                metric_tree = self._find_metric_tree(run_result.get("raw"))
                if metric_tree is not None:
                    snapshot["metrics_payload"] = metric_tree
                snapshot["run_query"] = {
                    "tool_name": "query_wandb_tool",
                    "arguments": run_result.get("arguments", {}),
                }

        if "count_weave_traces_tool" in names:
            snapshot["tool_results"]["count_weave_traces_tool"] = self._call_tool_with_variants(
                "count_weave_traces_tool",
                [
                    {"entity": entity, "project": project, "status": "failed"},
                    {"project": project, "status": "failed"},
                    {"entity": entity, "project": project},
                    {},
                ],
            )

        if "query_weave_traces_tool" in names:
            snapshot["tool_results"]["query_weave_traces_tool"] = self._call_tool_with_variants(
                "query_weave_traces_tool",
                [
                    {"entity": entity, "project": project, "status": "failed", "limit": 20},
                    {"project": project, "status": "failed", "limit": 20},
                    {"entity": entity, "project": project, "limit": 20},
                    {},
                ],
            )

        if "query_wandb_support_bot" in names:
            snapshot["tool_results"]["query_wandb_support_bot"] = self._call_tool_with_variants(
                "query_wandb_support_bot",
                [
                    {
                        "question": (
                            "How should I compare before/after model quality in W&B report "
                            "for iterative training loops?"
                        )
                    },
                    {},
                ],
            )

        if "create_wandb_report_tool" in names:
            report_title = f"MCP Loop Iteration {iter_idx} Report" if iter_idx else "MCP Loop Iteration Report"
            report_body = (
                f"Iteration={iter_idx}, overall_mae={overall_mae}, worst_axis={worst_axis}. "
                "This report was auto-created by the loop for before/after comparison."
            )
            snapshot["tool_results"]["create_wandb_report_tool"] = self._call_tool_with_variants(
                "create_wandb_report_tool",
                [
                    {
                        "entity": entity,
                        "project": project,
                        "title": report_title,
                        "description": report_body,
                    },
                    {
                        "project": project,
                        "title": report_title,
                        "description": report_body,
                    },
                    {
                        "title": report_title,
                        "description": report_body,
                    },
                ],
            )

        return snapshot