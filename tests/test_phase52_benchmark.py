import importlib.util
import json
from pathlib import Path
import pytest
import requests


SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_phase52.py"
SPEC = importlib.util.spec_from_file_location("benchmark_phase52", SCRIPT)
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_task_loading_has_five_tasks_per_role():
    assert [task["id"] for task in benchmark.load_tasks("coding")] == [
        "coding_01", "coding_02", "coding_03", "coding_04", "coding_05"]
    assert len(benchmark.load_tasks("review")) == 5


def test_safe_model_name_cannot_escape_result_directory():
    value = benchmark.safe_model_name("../../qwen:7b / token")
    assert value == "qwen-7b-token"
    assert "/" not in value and ".." not in value


def test_patch_extraction_accepts_fence_and_rejects_prose():
    patch = benchmark.extract_patch("Here\n```diff\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n```")
    assert patch.startswith("--- a/a.py")
    assert benchmark.extract_patch("I changed the file") is None


def test_coding_response_format_classification_is_deterministic():
    raw_diff = "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n"
    fenced = f"```diff\n{raw_diff}```"
    assert benchmark.classify_coding_response(raw_diff) == "unified_diff"
    assert benchmark.classify_coding_response(fenced) == "markdown_diff_fence"
    assert benchmark.classify_coding_response("I would change a.py") == "prose"
    assert benchmark.classify_coding_response("  \n") == "empty"
    assert benchmark.classify_coding_response("???") == "unknown"


def structured_response(edits):
    return {"ok":True, "content":json.dumps({"edits":edits}),
            "runtime_seconds":.1, "metrics":{}}


def test_structured_edit_generates_patch_and_passes_all_coding_01_tests(tmp_path):
    task = benchmark.load_tasks("coding")[0]
    response = structured_response([{
        "path":"router.py", "search":"return enabled[0] if enabled else None",
        "replace":"return None"}])
    result = benchmark.evaluate_coding_task(
        task, response, artifact_dir=tmp_path, coding_mode="structured_edit")
    assert result["coding_mode"] == "structured_edit"
    assert result["structured_response_parsed"] is True
    assert result["structured_schema_valid"] is True
    assert result["edits_count"] == 1 and result["edits_valid"] is True
    assert result["generated_patch"] and result["patch_syntax_valid"]
    assert result["patch_apply_success"] and result["compile_success"]
    assert result["visible_tests_success"] and result["hidden_tests_success"]
    assert result["tests_success"] and result["score"] == 1.0
    assert "return None" in (tmp_path / "coding_01.patch").read_text()


def test_replace_lines_prefixes_only_existing_base_indentation():
    source = "def run():\n        old()\n"
    updated, error = benchmark.apply_structured_edit(source, {
        "search":"old()",
        "replace_lines":["if condition:", "    new()", "final()"]})
    assert error is None
    assert updated == (
        "def run():\n"
        "        if condition:\n"
        "            new()\n"
        "        final()\n")


def test_replace_lines_does_not_repair_missing_relative_indentation():
    source = "def run():\n        old()\n"
    updated, error = benchmark.apply_structured_edit(source, {
        "search":"old()", "replace_lines":["if condition:", "new()"]})
    assert error is None
    assert "        new()" in updated
    with pytest.raises(IndentationError):
        compile(updated, "fixture.py", "exec")


@pytest.mark.parametrize("edit,error", [
    ({"path":"router.py", "search":"return None", "replace":"x",
      "replace_lines":["y"]}, "exactly one"),
    ({"path":"router.py", "search":"return None"}, "exactly one"),
    ({"path":"router.py", "search":"return None", "replace_lines":[]},
     "must not be empty"),
    ({"path":"router.py", "search":"return None", "replace_lines":["x\ny"]},
     "cannot contain newlines"),
])
def test_replace_lines_schema_is_strict(edit, error):
    parsed, message = benchmark.parse_structured_edits(json.dumps({"edits":[edit]}))
    assert parsed is None
    assert error in message


def test_replace_lines_rejects_partial_or_multiline_search():
    partial, partial_error = benchmark.apply_structured_edit(
        "    call(arg)\n", {"search":"call", "replace_lines":["other()"]})
    multiline, multiline_error = benchmark.apply_structured_edit(
        "    first()\n    second()\n",
        {"search":"first()\n    second()", "replace_lines":["other()"]})
    assert partial is None and "complete source line" in partial_error
    assert multiline is None and "one line" in multiline_error


def test_coding_02_replace_lines_generates_indentation_aware_patch(tmp_path):
    task = benchmark.load_tasks("coding")[1]
    response = structured_response([
        {"path":"history.py", "search":"self._items.append(item)",
         "replace_lines":["if len(self._items) >= self.history_limit:",
                          "    self._items.pop(0)", "self._items.append(item)"]},
        {"path":"history.py", "search":"self.history_limit = history_limit",
         "replace_lines":["if history_limit <= 0:",
                          "    raise ValueError(\"history_limit must be positive\")",
                          "self.history_limit = history_limit"]},
    ])
    result = benchmark.evaluate_coding_task(
        task, response, artifact_dir=tmp_path, coding_mode="structured_edit")
    patch = (tmp_path / "coding_02.patch").read_text()
    assert result["edits_valid"] and result["generated_patch"]
    assert result["compile_success"] and result["tests_success"]
    assert "+            self._items.pop(0)" in patch


def test_structured_edit_rejects_invalid_json_and_prose_without_execution():
    task = benchmark.load_tasks("coding")[0]
    for content in ("{not-json", "Please change the fallback to None"):
        result = benchmark.evaluate_coding_task(
            task, {"ok":True, "content":content}, coding_mode="structured_edit")
        assert result["structured_response_parsed"] is False
        assert result["structured_schema_valid"] is False
        assert result["generated_patch"] is False
        assert result["sandbox_execution_attempted"] is False


def test_structured_edit_rejects_disallowed_and_protected_paths():
    task = benchmark.load_tasks("coding")[0]
    cases = [
        ({"path":"other.py", "search":"x", "replace":"y"}, "path not allowed"),
        ({"path":"tests/test_router.py", "search":"WORKERS", "replace":"X"},
         "protected path"),
    ]
    for edit, error in cases:
        result = benchmark.evaluate_coding_task(
            task, structured_response([edit]), coding_mode="structured_edit")
        assert result["structured_schema_valid"] is True
        assert result["edits_valid"] is False and error in result["structured_error"]
        assert result["generated_patch"] is False


def test_structured_edit_requires_exactly_one_search_match():
    task = benchmark.load_tasks("coding")[0]
    for search, error in (("does not exist", "not found"), ("worker", "not unique")):
        result = benchmark.evaluate_coding_task(task, structured_response([{
            "path":"router.py", "search":search, "replace":"replacement"}]),
            coding_mode="structured_edit")
        assert result["edits_valid"] is False
        assert error in result["structured_error"]


def test_multiple_structured_edits_are_applied_deterministically():
    task = benchmark.load_tasks("coding")[0]
    response = structured_response([
        {"path":"router.py", "search":"return enabled[0] if enabled else None",
         "replace":"return None"},
        {"path":"router.py", "search":"# BUG: never substitute a worker with an incompatible capability.",
         "replace":"# No cross-capability fallback."},
    ])
    result = benchmark.evaluate_coding_task(task, response, coding_mode="structured_edit")
    assert result["edits_count"] == 2 and result["edits_valid"]
    assert result["tests_success"]


def test_structured_prompt_declares_json_contract_without_hidden_tests():
    task = benchmark.load_tasks("coding")[0]
    fixture = benchmark.BENCHMARK_ROOT / "fixtures" / task["fixture"]
    prompt = benchmark.prompt_for_coding(task, fixture, "structured_edit")
    assert "Return only valid JSON" in prompt
    assert '"search":"old_text"' in prompt
    assert '"replace_lines"' in prompt
    assert "test_vision_never_uses_coding" not in prompt


def test_coding_mode_cli_defaults_native_and_rejects_structured_review():
    assert benchmark.parse_args(["--role", "coding"]).coding_mode == "native_patch"
    assert benchmark.parse_args(
        ["--role", "coding", "--coding-mode", "structured_edit"]
    ).coding_mode == "structured_edit"
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--role", "review", "--coding-mode", "structured_edit"])


def test_granite_style_structured_edit_fails_hidden_tests():
    task = benchmark.load_tasks("coding")[0]
    result = benchmark.evaluate_coding_task(task, structured_response([{
        "path":"router.py", "search":"return enabled[0] if enabled else None",
        "replace":("compatible = [worker for worker in enabled "
                   "if worker.capability != \"reasoning\"]\n"
                   "    if compatible:\n        return compatible[0]\n    return None")}]),
        coding_mode="structured_edit")
    assert result["structured_schema_valid"] and result["edits_valid"]
    assert result["generated_patch"] and result["compile_success"]
    assert result["visible_tests_success"] is True
    assert result["hidden_tests_success"] is False
    assert result["tests_success"] is False
    assert result["score"] < .6


def test_qwen_style_structured_noop_fix_does_not_pass_tests():
    task = benchmark.load_tasks("coding")[0]
    result = benchmark.evaluate_coding_task(task, structured_response([{
        "path":"router.py", "search":"return exact[0]",
        "replace":"return exact[0]  # deterministic first match"}]),
        coding_mode="structured_edit")
    assert result["generated_patch"] and result["compile_success"]
    assert result["tests_success"] is False


def test_structured_edit_rejects_protected_test_modification():
    task = benchmark.load_tasks("coding")[0]
    result = benchmark.evaluate_coding_task(task, structured_response([{
        "path":"tests/test_router.py", "search":"WORKERS", "replace":"OTHERS"}]),
        coding_mode="structured_edit")
    assert result["edits_valid"] is False
    assert result["structured_error"].endswith("protected path")


def test_raw_coding_response_is_saved_exactly_when_no_patch_is_extracted(tmp_path):
    task = benchmark.load_tasks("coding")[0]
    raw = "Granite explanation with original spacing.\n\nNo patch was emitted.\n"
    result = benchmark.evaluate_coding_task(
        task, {"ok":True, "content":raw}, response_dir=tmp_path / "responses")
    saved = tmp_path / "responses" / "coding_01.txt"
    assert saved.read_text(encoding="utf-8") == raw
    assert result["patch_extracted"] is False
    assert result["response_length"] == len(raw)
    assert result["response_format"] == "prose"
    assert result["score"] == 0.0


def test_known_good_patch_applies_compiles_and_passes(tmp_path):
    task = benchmark.load_tasks("coding")[0]
    response = {"ok":True, "content":"""diff --git a/router.py b/router.py
--- a/router.py
+++ b/router.py
@@ -16,4 +16,4 @@ def select_worker(required_capability: str, workers: list[Worker]) -> Worker | None:
     if exact:
         return exact[0]
     # BUG: never substitute a worker with an incompatible capability.
-    return enabled[0] if enabled else None
+    return None
""", "runtime_seconds":.1, "metrics":{}}
    result = benchmark.evaluate_coding_task(task, response, artifact_dir=tmp_path)
    assert result["patch_extracted"] and result["patch_syntax_valid"]
    assert result["patch_apply_success"] and result["compile_success"]
    assert result["tests_success"] and result["score"] == 1.0
    assert result["visible_tests_success"] is True
    assert result["hidden_tests_success"] is True
    assert (tmp_path / "coding_01.patch").is_file()


def test_hidden_tests_are_not_in_prompt_and_are_installed_only_for_evaluation(tmp_path):
    task = benchmark.load_tasks("coding")[0]
    fixture = benchmark.BENCHMARK_ROOT / "fixtures" / task["fixture"]
    prompt = benchmark.prompt_for_coding(task, fixture)
    assert "test_vision_never_uses_coding" not in prompt
    work = tmp_path / "fixture"
    work.mkdir()
    destination = benchmark.install_hidden_tests(task, benchmark.BENCHMARK_ROOT, work)
    assert destination == work / ".benchmark_hidden_tests"
    assert (destination / "test_router_hidden.py").is_file()


def test_granite_style_cross_capability_fallback_fails_hidden_tests():
    task = benchmark.load_tasks("coding")[0]
    response = {"ok":True, "content":"""diff --git a/router.py b/router.py
--- a/router.py
+++ b/router.py
@@ -14,4 +14,7 @@ def select_worker(required_capability: str, workers: list[Worker]) -> Worker | None:
     if exact:
         return exact[0]
     # BUG: never substitute a worker with an incompatible capability.
-    return enabled[0] if enabled else None
+    compatible = [worker for worker in enabled if worker.capability != "reasoning"]
+    if compatible:
+        return compatible[0]
+    return None
""", "runtime_seconds":.1, "metrics":{}}
    result = benchmark.evaluate_coding_task(task, response)
    assert result["patch_syntax_valid"] and result["patch_apply_success"]
    assert result["compile_success"] is True
    assert result["visible_tests_success"] is True
    assert result["hidden_tests_success"] is False
    assert result["tests_success"] is False


def test_model_patch_cannot_replace_hidden_evaluator_tests():
    task = benchmark.load_tasks("coding")[0]
    response = {"ok":True, "content":"""diff --git a/.benchmark_hidden_tests/test_router_hidden.py b/.benchmark_hidden_tests/test_router_hidden.py
new file mode 100644
--- /dev/null
+++ b/.benchmark_hidden_tests/test_router_hidden.py
@@ -0,0 +1 @@
+def test_fake(): assert True
""", "runtime_seconds":.1, "metrics":{}}
    result = benchmark.evaluate_coding_task(task, response)
    assert result["patch_apply_success"] is True
    assert result["benchmark_invalid"] is True
    assert ".benchmark_hidden_tests/test_router_hidden.py" in result["changed_files"]
    assert result["score"] == 0.0


def test_invalid_patch_is_handled_without_crash():
    task = benchmark.load_tasks("coding")[0]
    result = benchmark.evaluate_coding_task(task, {"ok":True, "content":"not a patch"})
    assert not result["patch_extracted"] and not result["patch_syntax_valid"]
    assert not result["valid_patch"] and not result["patch_apply_success"]
    assert result["sandbox_execution_attempted"] is False
    assert result["sandboxed_execution"] is None
    assert result["score"] == 0.0


def test_malformed_unified_diff_is_diagnosed_before_apply():
    task = benchmark.load_tasks("coding")[0]
    response = {"ok":True, "content":"""diff --git a/router.py b/router.py
--- a/router.py
+++ b/router.py
@@ -10,7 +10,7 @@ def select_worker(required_capability: str, workers: list[Worker]) -> Worker | None:
     enabled = [worker for worker in workers if worker.enabled]
     exact = [worker for worker in enabled if worker.capability == required_capability]
     if exact:
-        return exact[0]
+        return next(iter(exact), None)
     # BUG: never substitute a worker with an incompatible capability.
     return enabled[0] if enabled else None
""", "runtime_seconds":.1, "metrics":{}}
    result = benchmark.evaluate_coding_task(task, response)
    assert result["response_received"] and result["patch_extracted"]
    assert result["patch_syntax_valid"] is False
    assert result["patch_apply_success"] is False
    assert result["patch_check_returncode"] != 0
    assert result["patch_check_stderr"]
    assert result["compile_success"] is False and result["tests_success"] is False
    assert result["sandbox_execution_attempted"] is False
    assert result["sandboxed_execution"] is None
    assert result["score"] == 0.0


def test_valid_but_semantically_wrong_patch_reaches_failing_tests():
    task = benchmark.load_tasks("coding")[0]
    response = {"ok":True, "content":"""diff --git a/router.py b/router.py
--- a/router.py
+++ b/router.py
@@ -11,7 +11,7 @@
 def select_worker(required_capability: str, workers: list[Worker]) -> Worker | None:
     enabled = [worker for worker in workers if worker.enabled]
     exact = [worker for worker in enabled if worker.capability == required_capability]
     if exact:
-        return exact[0]
+        return next(iter(exact), None)
     # BUG: never substitute a worker with an incompatible capability.
     return enabled[0] if enabled else None
""", "runtime_seconds":.1, "metrics":{}}
    result = benchmark.evaluate_coding_task(task, response)
    assert result["patch_syntax_valid"] and result["patch_apply_success"]
    assert result["compile_success"]
    assert result["tests_success"] is False


def test_delayed_ollama_unload_is_confirmed_without_fixed_sleep():
    model = "candidate:7b"
    states = iter([(model,), (model,), ()])
    current = {"time":0.0, "unloads":0}

    def unload(*_args, **_kwargs):
        current["unloads"] += 1
        return True

    def sleep(seconds):
        current["time"] += seconds

    result = benchmark.unload_model_bounded(
        model, "http://ollama", max_wait_seconds=1.0, poll_interval_seconds=.2,
        unload_fn=unload, residency_probe=lambda: next(states),
        clock=lambda: current["time"], sleeper=sleep)
    assert current["unloads"] == 1
    assert result["unload_requested"] and result["unload_request_success"]
    assert result["unload_confirmed"] is True
    assert result["model_still_resident"] is False
    assert result["unload_wait_seconds"] == .4


def test_ollama_unload_timeout_reports_resident_model():
    model = "candidate:7b"
    current = {"time":0.0}

    def sleep(seconds):
        current["time"] += seconds

    result = benchmark.unload_model_bounded(
        model, "http://ollama", max_wait_seconds=.5, poll_interval_seconds=.2,
        unload_fn=lambda *_args, **_kwargs: True,
        residency_probe=lambda: (model,), clock=lambda: current["time"], sleeper=sleep)
    assert result["unload_confirmed"] is False
    assert result["model_still_resident"] is True
    assert result["unload_wait_seconds"] == .5


def test_ollama_unload_request_failure_is_controlled():
    def fail(*_args, **_kwargs):
        raise requests.ConnectionError("offline")

    result = benchmark.unload_model_bounded(
        "candidate:7b", "http://ollama", unload_fn=fail,
        residency_probe=lambda: ())
    assert result["unload_request_success"] is False
    assert "ConnectionError" in result["unload_request_error"]
    assert result["unload_confirmed"] is True


def test_protected_file_hash_detects_test_manipulation(tmp_path):
    protected = tmp_path / "tests.py"; protected.write_text("assert True")
    original = benchmark.protected_hashes(tmp_path, ["tests.py"])
    protected.write_text("assert False")
    assert benchmark.protected_files_changed(tmp_path, original) == ["tests.py"]


def test_coding_score_weights_are_stable():
    assert benchmark.score_coding(response_received=True, patch_applied=True,
        compiled=True, tests_passed=False, discipline=True) == .4


def test_review_json_parsing_ignores_surrounding_thinking_text():
    parsed = benchmark.parse_review_json(
        "analysis omitted\n```json\n{\"verdict\":\"reject\",\"findings\":[]}\n```")
    assert parsed == {"verdict":"reject", "findings":[]}


def test_review_ground_truth_matches_concepts_within_one_finding():
    task = benchmark.load_tasks("review")[1]
    review = {"verdict":"reject", "findings":[{
        "severity":"high", "category":"concurrency", "summary":"Exception leaks lock",
        "reason":"This can deadlock", "recommendation":"release in finally"}]}
    result = benchmark.match_review_ground_truth(review, task["required_findings"])
    assert result["matched"] == ["lock_leak"] and result["score"] == 1.0


class FailingSession:
    def post(self, *_args, **_kwargs):
        raise requests.Timeout("slow")


def test_ollama_timeout_becomes_controlled_result():
    client = benchmark.OllamaBenchmarkClient("http://ollama/api/chat", session=FailingSession())
    result = client.generate("model", "prompt", timeout=.1, max_tokens=10, json_output=False)
    assert result["ok"] is False and "Timeout" in result["error"]


class FakeOllamaResponse:
    def __init__(self, payload, error=None):
        self.payload = payload
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise requests.HTTPError(self.error)

    def json(self):
        return self.payload


class CapturingSession:
    def __init__(self, response):
        self.response = response
        self.payload = None

    def post(self, _url, *, json, timeout):
        self.payload = json
        return self.response


@pytest.mark.parametrize("mode,expected", [("off", False), ("on", True)])
def test_ollama_request_sends_explicit_thinking_control(mode, expected):
    session = CapturingSession(FakeOllamaResponse({
        "message":{"content":"answer"}, "eval_count":1}))
    result = benchmark.OllamaBenchmarkClient("http://ollama", session=session).generate(
        "model", "prompt", timeout=1, max_tokens=10, json_output=False,
        thinking_mode=mode)
    assert session.payload["think"] is expected
    assert result["thinking_mode"] == mode
    assert result["benchmark_endpoint"] == "/"


def test_benchmark_records_chat_endpoint_without_url_credentials():
    session = CapturingSession(FakeOllamaResponse({
        "message":{"content":"answer"}, "eval_count":1, "done_reason":"stop"}))
    client = benchmark.OllamaBenchmarkClient(
        "http://user:password@ollama:11434/api/chat", session=session)
    result = client.generate("model", "prompt", timeout=1, max_tokens=10,
                             json_output=False)
    assert result["benchmark_endpoint"] == "/api/chat"
    assert "password" not in json.dumps(result)
    assert result["done_reason"] == "stop"


def test_separate_thinking_never_becomes_or_persists_as_final_response(tmp_path):
    secret_thinking = "private intermediate reasoning"
    final = '{"edits":[{"path":"router.py","search":"x","replace":"y"}]}'
    session = CapturingSession(FakeOllamaResponse({
        "message":{"thinking":secret_thinking, "content":final},
        "eval_count":20, "done_reason":"stop"}))
    response = benchmark.OllamaBenchmarkClient("http://ollama", session=session).generate(
        "model", "prompt", timeout=1, max_tokens=100, json_output=True,
        thinking_mode="on")
    assert response["content"] == final
    assert response["thinking_present"] is True
    assert response["thinking_length"] == len(secret_thinking)
    assert "thinking" not in response
    task = benchmark.load_tasks("coding")[0]
    result = benchmark.evaluate_coding_task(
        task, response, coding_mode="structured_edit", response_dir=tmp_path)
    assert (tmp_path / "coding_01.txt").read_text() == final
    assert secret_thinking not in (tmp_path / "coding_01.txt").read_text()
    assert result["structured_response_parsed"] is True


def test_empty_final_response_at_token_limit_is_generation_exhaustion():
    session = CapturingSession(FakeOllamaResponse({
        "message":{"content":"", "thinking":"used the whole budget"},
        "eval_count":1200, "done_reason":"length"}))
    response = benchmark.OllamaBenchmarkClient("http://ollama", session=session).generate(
        "model", "prompt", timeout=1, max_tokens=1200, json_output=True,
        thinking_mode="on")
    result = benchmark.evaluate_coding_task(
        benchmark.load_tasks("coding")[0], response, coding_mode="structured_edit")
    assert response["final_response_empty"] is True
    assert response["token_budget_exhausted"] is True
    assert result["response_received"] is False
    assert result["benchmark_outcome"] == "generation_exhausted_before_answer"
    assert result["structured_response_parsed"] is False
    assert result["compile_success"] is False and result["tests_success"] is False


def test_missing_thinking_field_and_unsupported_request_are_controlled():
    supported = CapturingSession(FakeOllamaResponse({
        "message":{"content":"answer"}, "eval_count":1}))
    result = benchmark.OllamaBenchmarkClient("http://ollama", session=supported).generate(
        "model", "prompt", timeout=1, max_tokens=10, json_output=False)
    assert result["thinking_present"] is False and result["thinking_length"] == 0

    unsupported = CapturingSession(FakeOllamaResponse({}, "400 think is unsupported"))
    failed = benchmark.OllamaBenchmarkClient("http://ollama", session=unsupported).generate(
        "model", "prompt", timeout=1, max_tokens=10, json_output=False)
    assert failed["ok"] is False
    assert "think is unsupported" in failed["error"]
    assert failed["final_response_empty"] is True


def test_native_and_structured_responses_keep_existing_evaluation_semantics():
    task = benchmark.load_tasks("coding")[0]
    native = {"ok":True, "content":"""diff --git a/router.py b/router.py
--- a/router.py
+++ b/router.py
@@ -16,4 +16,4 @@ def select_worker(required_capability: str, workers: list[Worker]) -> Worker | None:
     if exact:
         return exact[0]
     # BUG: never substitute a worker with an incompatible capability.
-    return enabled[0] if enabled else None
+    return None
""", "thinking_mode":"off"}
    structured = structured_response([{"path":"router.py",
        "search":"return enabled[0] if enabled else None", "replace":"return None"}])
    structured["thinking_mode"] = "off"
    assert benchmark.evaluate_coding_task(task, native)["tests_success"] is True
    assert benchmark.evaluate_coding_task(
        task, structured, coding_mode="structured_edit")["tests_success"] is True


def test_gpu_metrics_handles_provider_failure():
    class Provider:
        def snapshot(self):
            from avacore.model.router import ResourceSnapshot
            return ResourceSnapshot()
    assert benchmark.gpu_metrics(Provider())["gpu_used_mb"] is None


def test_comparison_results_are_serializable(tmp_path):
    summary = {"model":"test", "role":"coding", "patches_applied":1,
        "compiled":1, "tasks_passed":1, "tasks_total":1, "score":1.0,
        "mean_runtime_seconds":.1, "gpu_used_mb":100}
    benchmark.write_comparison(tmp_path, [summary])
    assert json.loads((tmp_path / "comparison.json").read_text())["models"][0]["model"] == "test"
    assert "| test |" in (tmp_path / "comparison.md").read_text()


def test_structured_comparison_reports_mode_and_metrics(tmp_path):
    summary = {"model":"test", "role":"coding", "coding_mode":"structured_edit",
        "structured_parsed":1, "valid_edits":1, "patches_generated":1, "compiled":1,
        "tasks_passed":1, "tasks_total":1, "score":1.0,
        "mean_runtime_seconds":.1, "gpu_used_mb":100}
    benchmark.write_comparison(tmp_path, [summary])
    payload = json.loads((tmp_path / "comparison.json").read_text())
    markdown = (tmp_path / "comparison.md").read_text()
    assert payload["coding_mode"] == "structured_edit"
    assert "Mode: structured_edit" in markdown
    assert "Valid edits" in markdown
