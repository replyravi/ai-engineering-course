"""
tests/test_llm.py - checks that llm.py picks the right provider from .env.

These tests only BUILD the model objects (no question is sent), so they run
offline with a dummy key. The last four prove that the ONE .env file at the
repo root is found from inside an episode folder.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import llm as llm_module


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    # Start every test with no provider settings and no keys.
    for name in [
        "LLM_PROVIDER", "LLM_MODEL", "GEMINI_API_KEY", "GOOGLE_API_KEY",
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "EMBEDDINGS_PROVIDER", "EMBEDDINGS_MODEL",
    ]:
        monkeypatch.delenv(name, raising=False)


def test_default_is_gemini(monkeypatch):
    pytest.importorskip("langchain_google_genai")  # skipped in the CrewAI-only environment
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-key-for-tests")
    model = llm_module.get_llm()
    assert type(model).__name__ == "ChatGoogleGenerativeAI"
    assert model.model == llm_module.DEFAULT_MODELS["gemini"]


def test_missing_key_gives_beginner_friendly_help():
    with pytest.raises(llm_module.MissingAPIKeyError) as error:
        llm_module.get_llm()
    assert "https://aistudio.google.com/apikey" in str(error.value)


def test_placeholder_key_counts_as_missing(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "paste-your-key-here")
    with pytest.raises(llm_module.MissingAPIKeyError):
        llm_module.get_llm()


@pytest.mark.parametrize(
    "provider, key_name, package, class_name",
    [
        ("openai", "OPENAI_API_KEY", "langchain_openai", "ChatOpenAI"),
        ("anthropic", "ANTHROPIC_API_KEY", "langchain_anthropic", "ChatAnthropic"),
        ("ollama", None, "langchain_ollama", "ChatOllama"),
    ],
)
def test_one_line_switch(monkeypatch, provider, key_name, package, class_name):
    pytest.importorskip(package)
    monkeypatch.setenv("LLM_PROVIDER", provider)
    if key_name:
        monkeypatch.setenv(key_name, "dummy-key-for-tests")
    assert type(llm_module.get_llm()).__name__ == class_name


def test_llm_model_overrides_the_default(monkeypatch):
    pytest.importorskip("langchain_google_genai")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-key-for-tests")
    monkeypatch.setenv("LLM_MODEL", "gemini-3.8-flash")
    assert llm_module.get_llm().model == "gemini-3.8-flash"


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "my-own-ai")
    with pytest.raises(ValueError, match="not supported"):
        llm_module.get_llm()


def test_crewai_llm_uses_provider_prefix(monkeypatch):
    pytest.importorskip("crewai")  # only installed for the CrewAI episodes
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-key-for-tests")
    crew_llm = llm_module.get_crewai_llm()
    assert crew_llm.model == llm_module.DEFAULT_MODELS["gemini"]
    assert crew_llm.provider == "gemini"


# ---------------------------------------------------------------------------
# ONE .env at the repo root, found from inside any episode folder
# ---------------------------------------------------------------------------
SHOW_SETTINGS = "import llm, os; print(llm.ENV_FILE); print(os.environ.get('LLM_MODEL', 'not set'))"


def make_fake_repo(tmp_path):
    """Build ai-engineering-course/.env + ep99-demo/llm.py, like a viewer's clone."""
    repo = tmp_path / "ai-engineering-course"
    episode = repo / "ep99-demo"
    episode.mkdir(parents=True)
    (repo / ".env").write_text("LLM_MODEL=model-from-the-root-env\n", encoding="utf-8")
    shutil.copy(Path(llm_module.__file__), episode / "llm.py")
    (episode / "show_settings.py").write_text(SHOW_SETTINGS.replace("; ", "\n") + "\n", encoding="utf-8")
    return repo, episode


def run_python(args, cwd):
    """Run a fresh Python (like a viewer typing 'python ...') with no course settings in its environment."""
    env = {k: v for k, v in os.environ.items() if k not in ("LLM_MODEL", "LLM_PROVIDER", "PYTHONPATH")}
    result = subprocess.run([sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines()


def same_file(a, b):
    return Path(a).resolve() == Path(b).resolve()


def test_find_env_file_walks_up_from_the_episode_folder(tmp_path, monkeypatch):
    repo, episode = make_fake_repo(tmp_path)
    monkeypatch.chdir(episode)
    assert same_file(llm_module.find_env_file(), repo / ".env")


def test_root_env_is_loaded_when_a_script_runs_inside_an_episode_folder(tmp_path):
    repo, episode = make_fake_repo(tmp_path)
    env_file, model = run_python(["show_settings.py"], cwd=episode)   # cd ep99-demo; python show_settings.py
    assert same_file(env_file, repo / ".env")
    assert model == "model-from-the-root-env"


def test_root_env_is_loaded_when_run_from_the_repo_root(tmp_path):
    repo, episode = make_fake_repo(tmp_path)
    env_file, model = run_python(["ep99-demo/show_settings.py"], cwd=repo)   # python ep99-demo/show_settings.py
    assert same_file(env_file, repo / ".env")
    assert model == "model-from-the-root-env"


def test_root_env_is_found_from_llm_py_when_run_from_somewhere_else(tmp_path):
    repo, episode = make_fake_repo(tmp_path)
    elsewhere = tmp_path / "somewhere-else"
    elsewhere.mkdir()
    env_file, model = run_python([str(episode / "show_settings.py")], cwd=elsewhere)
    assert same_file(env_file, repo / ".env")
    assert model == "model-from-the-root-env"
