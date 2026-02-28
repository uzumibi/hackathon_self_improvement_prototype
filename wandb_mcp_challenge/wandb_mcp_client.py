import json
import os
from typing import Any, Dict, List, Optional, Tuple

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
    def _extract_structured_result(raw: Dict[str, Any]) -> Any:
        result = raw.get("result", {})
        if not isinstance(result, dict):
            return None
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            if "result" in structured:
                return structured.get("result")
            return structured
        return None

    @staticmethod
    def _extract_first_text_content(raw: Dict[str, Any]) -> Optional[str]:
        result = raw.get("result", {})
        if not isinstance(result, dict):
            return None
        content = result.get("content", [])
        if not isinstance(content, list):
            return None
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    return text
        return None

    @staticmethod
    def _extract_tool_error(raw: Dict[str, Any]) -> Optional[str]:
        result = raw.get("result", {})
        if not isinstance(result, dict):
            return None
        if result.get("isError"):
            text = WandbMCPClient._extract_first_text_content(raw)
            return text or "tool returned isError=true"
        return None

    @staticmethod
    def _maybe_parse_json_text(text: Optional[str]) -> Any:
        if not text or not isinstance(text, str):
            return None
        stripped = text.strip()
        if not stripped:
            return None
        if not (stripped.startswith("{") or stripped.startswith("[")):
            return None
        try:
            return json.loads(stripped)
        except Exception:
            return None

    @staticmethod
    def _find_metric_tree(obj: Any) -> Optional[Dict[str, Any]]:
        if isinstance(obj, dict):
            has_target_key = any(
                k in obj
                for k in [
                    "overall_mae",
                    "within_tolerance_rate",
                    "mae_per_axis",
                    "summary",
                    "metrics",
                    "summaryMetrics",
                ]
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
                tool_error = self._extract_tool_error(raw)
                structured = self._extract_structured_result(raw)
                text = self._extract_first_text_content(raw)
                text_json = self._maybe_parse_json_text(text)

                if tool_error:
                    attempted.append(
                        {
                            "tool_name": tool_name,
                            "arguments": clean_args,
                            "error": tool_error,
                        }
                    )
                    continue

                return {
                    "ok": True,
                    "tool_name": tool_name,
                    "arguments": clean_args,
                    "raw": raw,
                    "structured_result": structured,
                    "text_result": text,
                    "text_json": text_json,
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

    @staticmethod
    def _parse_entity_projects(project_result: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        if not project_result.get("ok"):
            return {}
        structured = project_result.get("structured_result")
        if isinstance(structured, dict):
            parsed: Dict[str, List[Dict[str, Any]]] = {}
            for entity, items in structured.items():
                if isinstance(entity, str) and isinstance(items, list):
                    parsed[entity] = [p for p in items if isinstance(p, dict)]
            if parsed:
                return parsed

        text_json = project_result.get("text_json")
        if isinstance(text_json, dict):
            parsed = {}
            for entity, items in text_json.items():
                if isinstance(entity, str) and isinstance(items, list):
                    parsed[entity] = [p for p in items if isinstance(p, dict)]
            return parsed
        return {}

    @staticmethod
    def _select_entity_and_projects(
        entity_projects: Dict[str, List[Dict[str, Any]]],
        preferred_entity: Optional[str],
        model_project: str,
        weave_project: str,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        if preferred_entity and preferred_entity in entity_projects:
            return preferred_entity, {
                "model_project": model_project,
                "weave_project": weave_project,
            }

        for entity_name, projects in entity_projects.items():
            names = {p.get("name") for p in projects if isinstance(p, dict)}
            if model_project in names or weave_project in names:
                return entity_name, {
                    "model_project": model_project,
                    "weave_project": weave_project,
                }

        if entity_projects:
            first_entity = next(iter(entity_projects.keys()))
            return first_entity, {
                "model_project": model_project,
                "weave_project": weave_project,
            }

        return preferred_entity, {
            "model_project": model_project,
            "weave_project": weave_project,
        }

    def safe_fetch_latest_metrics(
        self,
        entity: Optional[str] = None,
        project: Optional[str] = None,
        loop_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Best-effort MCP inspection for self-improvement loop.
        Uses tool-specific argument schemas and narrows scope to relevant projects.
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

        model_project = project or "music-params-mcp-loop"
        weave_project = lc.get("weave_project_name") or "music-params-eval"

        snapshot: Dict[str, Any] = {
            "mcp_ok": True,
            "tools_discovered": names,
            "loop_tool_policy": {
                "required_tools": required_tools,
                "available_tools": available_tools,
                "scoped_projects": {
                    "model_project": model_project,
                    "weave_project": weave_project,
                },
                "message": "Use schema-correct tool calls and focus on music-params-* projects.",
            },
            "tool_results": {},
        }

        selected_entity = entity
        parsed_entity_projects: Dict[str, List[Dict[str, Any]]] = {}

        if "query_wandb_entity_projects" in names:
            proj_result = self._call_tool_with_variants(
                "query_wandb_entity_projects",
                [
                    {"entity": entity},
                    {},
                ],
            )
            snapshot["tool_results"]["query_wandb_entity_projects"] = proj_result
            parsed_entity_projects = self._parse_entity_projects(proj_result)
            selected_entity, selected_projects = self._select_entity_and_projects(
                parsed_entity_projects,
                preferred_entity=entity,
                model_project=model_project,
                weave_project=weave_project,
            )
            snapshot["project_resolution"] = {
                "selected_entity": selected_entity,
                "selected_projects": selected_projects,
                "entity_project_count": {k: len(v) for k, v in parsed_entity_projects.items()},
            }

        if "query_wandb_tool" in names and selected_entity:
            runs_query = """
query RecentRuns($entity: String!, $project: String!, $limit: Int) {
  project(name: $project, entityName: $entity) {
    id
    name
    runCount
    runs(first: $limit, order: "-createdAt") {
      edges {
        node {
          id
          name
          displayName
          state
          createdAt
          summaryMetrics
        }
      }
      pageInfo {
        endCursor
        hasNextPage
      }
    }
  }
}
""".strip()

            run_result = self._call_tool_with_variants(
                "query_wandb_tool",
                [
                    {
                        "query": runs_query,
                        "variables": {
                            "entity": selected_entity,
                            "project": model_project,
                            "limit": 8,
                        },
                        "max_items": 20,
                        "items_per_page": 8,
                    }
                ],
            )
            snapshot["tool_results"]["query_wandb_tool"] = run_result
            if run_result.get("ok"):
                metric_tree = self._find_metric_tree(
                    run_result.get("structured_result")
                    if run_result.get("structured_result") is not None
                    else run_result.get("raw")
                )
                if metric_tree is not None:
                    snapshot["metrics_payload"] = metric_tree
                snapshot["run_query"] = {
                    "tool_name": "query_wandb_tool",
                    "arguments": run_result.get("arguments", {}),
                }

        if "count_weave_traces_tool" in names and selected_entity:
            snapshot["tool_results"]["count_weave_traces_tool"] = self._call_tool_with_variants(
                "count_weave_traces_tool",
                [
                    {
                        "entity_name": selected_entity,
                        "project_name": weave_project,
                        "filters": {"status": "error", "trace_roots_only": True},
                    }
                ],
            )

        if "query_weave_traces_tool" in names and selected_entity:
            snapshot["tool_results"]["query_weave_traces_tool"] = self._call_tool_with_variants(
                "query_weave_traces_tool",
                [
                    {
                        "entity_name": selected_entity,
                        "project_name": weave_project,
                        "filters": {"status": "error", "trace_roots_only": True},
                        "columns": [
                            "id",
                            "trace_id",
                            "op_name",
                            "display_name",
                            "started_at",
                            "ended_at",
                            "status",
                            "latency_ms",
                            "exception",
                            "inputs",
                            "output",
                        ],
                        "limit": 10,
                        "return_full_data": False,
                        "truncate_length": 300,
                    }
                ],
            )

        if "query_wandb_support_bot" in names:
            snapshot["tool_results"]["query_wandb_support_bot"] = self._call_tool_with_variants(
                "query_wandb_support_bot",
                [
                    {
                        "question": (
                            f"In project {model_project}, what is the best way to compare "
                            "before/after loss and accuracy between recent runs in a report?"
                        )
                    }
                ],
            )

        if "create_wandb_report_tool" in names and selected_entity:
            report_title = f"MCP Loop Iteration {iter_idx} Report" if iter_idx else "MCP Loop Iteration Report"
            markdown_report_text = f"""# MCP Loop Iteration {iter_idx if iter_idx is not None else "-"}

## Scope
- Entity: `{selected_entity}`
- Model project: `{model_project}`
- Weave project: `{weave_project}`

## Evaluation Snapshot
- overall_mae: `{overall_mae}`
- worst_axis: `{worst_axis}`

## Notes
This report is auto-created for before/after comparison in the self-improvement loop.
"""

            snapshot["tool_results"]["create_wandb_report_tool"] = self._call_tool_with_variants(
                "create_wandb_report_tool",
                [
                    {
                        "entity_name": selected_entity,
                        "project_name": model_project,
                        "title": report_title,
                        "description": (
                            f"Iteration={iter_idx}, overall_mae={overall_mae}, worst_axis={worst_axis}"
                        ),
                        "markdown_report_text": markdown_report_text,
                    }
                ],
            )

        return snapshot