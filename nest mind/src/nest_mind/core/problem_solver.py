"""
Day 4 implementation: Problem-Solving Logic & AI Integration

Drop this file into:
    src/nest_mind/core/problem_solver.py

It expects an existing ConversationTree implementation (from Day 3).
It will try to call `conversation_tree.add_child_node()`; if not present,
it falls back to `conversation_tree.add_node()` or `conversation_tree.create_node()`.
"""

import os
import time
import typing as t

try:
    import openai
except Exception:
    openai = None  # openai optional if using local model

# Constants (tweakable)
DEFAULT_MODEL = "gpt-4o-mini"
MAX_RECURSION_DEPTH = 2      # how deep to recursively decompose tasks
MAX_RETRIES = 3
BASE_BACKOFF = 1.0           # seconds


class AIClient:
    """Small wrapper around OpenAI calls with retries and basic fallback support."""

    def __init__(self, api_key: t.Optional[str] = None, model: str = DEFAULT_MODEL, local_model_fn: t.Optional[t.Callable] = None):
        """
        local_model_fn: optional callable(prompt:str) -> str to use instead of OpenAI.
        """
        self.model = model
        self.local_model_fn = local_model_fn

        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")

        self.api_key = api_key
        if api_key and openai:
            openai.api_key = api_key

    def call(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.2) -> str:
        """Call the configured model. Retries on transient exceptions."""
        # If a local model function supplied, use it.
        if self.local_model_fn:
            try:
                return self.local_model_fn(prompt)
            except Exception as e:
                return f"[local-model-failed] {e}"

        # Otherwise use OpenAI (if available)
        if not (openai and self.api_key):
            # Fallback heuristic: return prompt-based simple echo / split suggestion
            return self._fallback_response(prompt)

        backoff = BASE_BACKOFF
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                text = resp.choices[0].message["content"].strip()
                return text
            except Exception as e:
                # Log and retry with exponential backoff
                print(f"[AIClient] attempt {attempt} failed: {e}. backing off {backoff}s")
                time.sleep(backoff)
                backoff *= 2
        return "[AIClient] failed after retries"

    def _fallback_response(self, prompt: str) -> str:
        # Simple fallback: if asked to break down, try to split by sentences; otherwise echo.
        if "Break this problem" in prompt or "Break this" in prompt or "split" in prompt.lower():
            # attempt naive sentence split
            parts = [p.strip() for p in prompt.split("\n") if p.strip()]
            # return a few numbered items
            return "\n".join(f"- {p[:120]}" for p in parts[:4]) or "1. Make plan\n2. Execute\n3. Test"
        return "Fallback: " + (prompt[:400] + ("..." if len(prompt) > 400 else ""))


class ProblemSolver:
    """
    Coordinates decomposition, routing into ConversationTree, solving, iterative refinement,
    and aggregation of partial solutions into a final answer.
    """

    def __init__(
        self,
        ai_client: AIClient,
        min_subtasks: int = 3,
        max_subtasks: int = 6,
        complexity_word_threshold: int = 40,
    ):
        self.ai = ai_client
        self.min_subtasks = min_subtasks
        self.max_subtasks = max_subtasks
        self.complexity_word_threshold = complexity_word_threshold

    # -------------------------
    # Decomposition & heuristics
    # -------------------------
    def decompose_problem(self, problem: str) -> t.List[str]:
        """
        Ask the AI to break a problem into smaller tasks.
        Returns a list of subtasks (strings).
        """
        prompt = (
            f"Break this problem into {self.min_subtasks}-{self.max_subtasks} "
            f"concise, actionable sub-tasks (one per line). Problem:\n\n{problem}\n\n"
            "Return each subtask as a short sentence starting with '-' or a new line."
        )
        result = self.ai.call(prompt)
        return self._parse_list_like_response(result)

    def _parse_list_like_response(self, text: str) -> t.List[str]:
        lines = []
        for raw in text.splitlines():
            s = raw.strip()
            if not s:
                continue
            # remove leading list markers
            if s.startswith("-") or s.startswith("*") or s[0].isdigit() and (s[1] in ". "):
                # remove "- " or "1." etc
                s = s.lstrip("-*0123456789. ").strip()
            lines.append(s)
        # fallback: if no lines, try sentence splitting
        if not lines:
            splits = [p.strip() for p in text.split(".") if p.strip()]
            if splits:
                lines = splits[: self.max_subtasks]
        return lines

    def is_complex(self, text: str) -> bool:
        """Simple heuristic to decide whether a text is 'complex' and needs further decomposition"""
        word_count = len(text.split())
        return word_count >= self.complexity_word_threshold

    # -------------------------
    # Routing into ConversationTree
    # -------------------------
    def _add_child_to_tree(self, conversation_tree, parent_node_id: t.Optional[str], title: str):
        """
        Attempts to add a new child node to ConversationTree.
        Tries a few common method names used in different implementations.
        Returns the new node id or node object (whatever the tree returns) or None.
        """
        # try common method names
        for method_name in ("add_child_node", "add_node", "create_node", "add_child"):
            method = getattr(conversation_tree, method_name, None)
            if callable(method):
                try:
                    # try to call with (title) or (title, parent_id)
                    try:
                        if parent_node_id is not None:
                            return method(title, parent_node_id)
                    except TypeError:
                        pass
                    # fallback to calling with only title
                    return method(title)
                except Exception as e:
                    print(f"[ProblemSolver] tree.{method_name} raised: {e}")
        print("[ProblemSolver] could not add node; ConversationTree API mismatch")
        return None

    def route_subproblems(self, conversation_tree, parent_node_id: t.Optional[str], sub_tasks: t.List[str]) -> t.List[t.Any]:
        """
        Create child nodes for each subtask. Returns list of created node ids/objects.
        parent_node_id: optional — attach under a given node
        """
        created = []
        for tstr in sub_tasks:
            node = self._add_child_to_tree(conversation_tree, parent_node_id, tstr)
            created.append(node)
        return created

    # -------------------------
    # Solve / refine / aggregate
    # -------------------------
    def _solve_single(self, subtask: str) -> str:
        """
        Solve one subtask by calling the AI. Returns the text answer.
        """
        prompt = f"Solve this task in detail. Provide steps, reasoning, and a concise final recommendation:\n\n{subtask}"
        return self.ai.call(prompt)

    def solve(
        self,
        conversation_tree,
        root_problem: str,
        root_node_id: t.Optional[str] = None,
        max_depth: int = MAX_RECURSION_DEPTH,
    ) -> str:
        """
        Complete pipeline:
          1) decompose root problem,
          2) route subtasks to the ConversationTree,
          3) for each subtask: solve it; if it's still complex and depth allows, recursively decompose & solve,
          4) aggregate partial solutions into a final combined answer.
        Returns final aggregated string.
        """
        # 1) Decompose root problem
        subtasks = self.decompose_problem(root_problem)
        if not subtasks:
            # fallback: treat whole problem as a single task
            subtasks = [root_problem]

        # 2) Route to tree (create nodes)
        created_nodes = self.route_subproblems(conversation_tree, root_node_id, subtasks)

        # 3) Solve each subtask (with optional recursive decomposition)
        partial_solutions = []
        for idx, sub in enumerate(subtasks):
            print(f"[ProblemSolver] Solving subtask {idx+1}/{len(subtasks)}: {sub}")

            # if subtask is complex and we can recurse, break it down further
            if max_depth > 0 and self.is_complex(sub):
                print(f"[ProblemSolver] Subtask appears complex (words={len(sub.split())}). Recursing...")
                # create a child node for recursion as parent (if tree returned node id/object use it)
                sub_parent_id = None
                if idx < len(created_nodes):
                    sub_parent_id = created_nodes[idx]
                # recursive decomposition/solve
                deeper_solution = self.solve(conversation_tree, sub, root_node_id=sub_parent_id, max_depth=max_depth - 1)
                partial_solutions.append(f"Subtask: {sub}\nSolution (aggregated recursion):\n{deeper_solution}\n")
            else:
                # solve directly
                try:
                    answer = self._solve_single(sub)
                except Exception as e:
                    answer = f"[Error solving subtask] {e}"
                partial_solutions.append(f"Subtask: {sub}\nSolution:\n{answer}\n")

        # 4) Aggregate partial solutions into final output
        aggregate_prompt = (
            "You are given partial solutions to different subtasks. Combine them into a single, "
            "coherent, step-by-step final solution. For each subtask, mention the main recommendation, "
            "possible risks, and a short action checklist.\n\n"
            + "\n\n".join(partial_solutions)
        )
        final_answer = self.ai.call(aggregate_prompt, max_tokens=1500, temperature=0.1)
        return final_answer
