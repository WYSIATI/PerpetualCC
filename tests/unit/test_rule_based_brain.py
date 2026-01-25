"""Unit tests for rule-based brain implementation."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pytest

from perpetualcc.brain.base import PermissionContext, QuestionContext
from perpetualcc.brain.rule_based import (
    PermissionPattern,
    QuestionPattern,
    RuleBasedBrain,
    RuleBasedConfig,
    default_permission_patterns,
    default_question_patterns,
)


@pytest.fixture
def temp_project() -> Path:
    """Create a temporary project directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project = Path(tmpdir)
        (project / "src").mkdir()
        (project / "tests").mkdir()
        yield project


@pytest.fixture
def brain() -> RuleBasedBrain:
    """Create a rule-based brain with default configuration."""
    return RuleBasedBrain()


@pytest.fixture
def question_context(temp_project: Path) -> QuestionContext:
    """Create a question context for testing."""
    return QuestionContext(
        project_path=str(temp_project),
        question="Test question",
        options=[{"label": "Yes"}, {"label": "No"}],
    )


@pytest.fixture
def permission_context(temp_project: Path) -> PermissionContext:
    """Create a permission context for testing."""
    return PermissionContext(project_path=str(temp_project))


class TestQuestionPattern:
    """Tests for QuestionPattern dataclass."""

    def test_pattern_is_immutable(self):
        """QuestionPattern should be immutable."""
        pattern = QuestionPattern(
            pattern=re.compile(r"test"),
            answer="Yes",
            confidence=0.8,
            reasoning="Test",
        )
        with pytest.raises(AttributeError):
            pattern.answer = "No"

    def test_pattern_fields(self):
        """QuestionPattern should have all required fields."""
        pattern = QuestionPattern(
            pattern=re.compile(r"proceed\?"),
            answer="Yes",
            confidence=0.85,
            reasoning="Standard confirmation",
        )
        assert pattern.pattern.pattern == r"proceed\?"
        assert pattern.answer == "Yes"
        assert pattern.confidence == 0.85
        assert pattern.reasoning == "Standard confirmation"


class TestPermissionPattern:
    """Tests for PermissionPattern dataclass."""

    def test_pattern_is_immutable(self):
        """PermissionPattern should be immutable."""
        pattern = PermissionPattern(
            tool_name="Bash",
            input_pattern=re.compile(r"git"),
            approve=True,
            confidence=0.8,
            reasoning="Test",
        )
        with pytest.raises(AttributeError):
            pattern.approve = False

    def test_pattern_with_none_tool(self):
        """PermissionPattern can have None tool_name to match any tool."""
        pattern = PermissionPattern(
            tool_name=None,
            input_pattern=re.compile(r"test"),
            approve=True,
            confidence=0.7,
            reasoning="Matches any tool",
        )
        assert pattern.tool_name is None

    def test_pattern_with_none_input_pattern(self):
        """PermissionPattern can have None input_pattern to match any input."""
        pattern = PermissionPattern(
            tool_name="Task",
            input_pattern=None,
            approve=True,
            confidence=0.75,
            reasoning="Matches any task input",
        )
        assert pattern.input_pattern is None


class TestRuleBasedConfig:
    """Tests for RuleBasedConfig dataclass."""

    def test_config_is_immutable(self):
        """RuleBasedConfig should be immutable."""
        config = RuleBasedConfig()
        with pytest.raises(AttributeError):
            config.confidence_threshold = 0.5

    def test_default_config(self):
        """Default config should have empty patterns."""
        config = RuleBasedConfig()
        assert config.question_patterns == ()
        assert config.permission_patterns == ()
        assert config.confidence_threshold == 0.7

    def test_custom_config(self):
        """Config should accept custom patterns."""
        patterns = (
            QuestionPattern(
                pattern=re.compile(r"custom"),
                answer="Custom",
                confidence=0.9,
                reasoning="Custom pattern",
            ),
        )
        config = RuleBasedConfig(question_patterns=patterns, confidence_threshold=0.8)
        assert len(config.question_patterns) == 1
        assert config.confidence_threshold == 0.8


class TestDefaultPatterns:
    """Tests for default pattern functions."""

    def test_default_question_patterns_not_empty(self):
        """Default question patterns should not be empty."""
        patterns = default_question_patterns()
        assert len(patterns) > 0

    def test_default_permission_patterns_not_empty(self):
        """Default permission patterns should not be empty."""
        patterns = default_permission_patterns()
        assert len(patterns) > 0

    def test_default_patterns_are_tuples(self):
        """Default patterns should be tuples (immutable)."""
        assert isinstance(default_question_patterns(), tuple)
        assert isinstance(default_permission_patterns(), tuple)


class TestRuleBasedBrainInit:
    """Tests for RuleBasedBrain initialization."""

    def test_default_initialization(self):
        """Brain should initialize with default patterns."""
        brain = RuleBasedBrain()
        assert len(brain.config.question_patterns) > 0
        assert len(brain.config.permission_patterns) > 0

    def test_custom_config(self):
        """Brain should accept custom configuration."""
        custom_config = RuleBasedConfig(confidence_threshold=0.9)
        brain = RuleBasedBrain(config=custom_config)
        assert brain.get_confidence_threshold() == 0.9

    def test_config_property(self):
        """Brain should expose config as property."""
        brain = RuleBasedBrain()
        config = brain.config
        assert isinstance(config, RuleBasedConfig)


class TestAnswerQuestionConfirmations:
    """Tests for answering confirmation questions."""

    @pytest.mark.asyncio
    async def test_proceed_question(self, brain: RuleBasedBrain, question_context: QuestionContext):
        """'proceed?' should be answered with Yes."""
        context = QuestionContext(
            project_path=question_context.project_path,
            question="proceed?",
            options=[{"label": "Yes"}, {"label": "No"}],
        )
        answer = await brain.answer_question("proceed?", [], context)
        assert answer.selected == "Yes"
        assert answer.confidence >= 0.8

    @pytest.mark.asyncio
    async def test_continue_question(
        self, brain: RuleBasedBrain, question_context: QuestionContext
    ):
        """'continue?' should be answered with Yes."""
        context = QuestionContext(
            project_path=question_context.project_path,
            question="continue?",
            options=[{"label": "Yes"}, {"label": "No"}],
        )
        answer = await brain.answer_question("continue?", [], context)
        assert answer.selected == "Yes"
        assert answer.confidence >= 0.8

    @pytest.mark.asyncio
    async def test_should_i_proceed(self, brain: RuleBasedBrain, question_context: QuestionContext):
        """'Should I proceed?' should be answered with Yes."""
        context = QuestionContext(
            project_path=question_context.project_path,
            question="Should I proceed?",
            options=[],
        )
        answer = await brain.answer_question("Should I proceed?", [], context)
        assert answer.selected == "Yes"
        assert answer.confidence >= 0.8

    @pytest.mark.asyncio
    async def test_can_i_continue(self, brain: RuleBasedBrain, question_context: QuestionContext):
        """'Can I continue?' should be answered with Yes."""
        context = QuestionContext(
            project_path=question_context.project_path,
            question="Can I continue?",
            options=[],
        )
        answer = await brain.answer_question("Can I continue?", [], context)
        assert answer.selected == "Yes"
        assert answer.confidence >= 0.8

    @pytest.mark.asyncio
    async def test_is_this_ok(self, brain: RuleBasedBrain, question_context: QuestionContext):
        """'Is this ok?' style questions should be answered with Yes."""
        context = QuestionContext(
            project_path=question_context.project_path,
            question="Does this look ok?",
            options=[],
        )
        answer = await brain.answer_question("Does this look ok?", [], context)
        assert answer.selected == "Yes"
        assert answer.confidence >= 0.7

    @pytest.mark.asyncio
    async def test_ready_to_start(self, brain: RuleBasedBrain, question_context: QuestionContext):
        """'Ready to start?' should be answered with Yes."""
        context = QuestionContext(
            project_path=question_context.project_path,
            question="Ready to start?",
            options=[],
        )
        answer = await brain.answer_question("Ready to start?", [], context)
        assert answer.selected == "Yes"
        assert answer.confidence >= 0.8


class TestAnswerQuestionOperations:
    """Tests for answering standard operation questions."""

    @pytest.mark.asyncio
    async def test_run_tests_question(
        self, brain: RuleBasedBrain, question_context: QuestionContext
    ):
        """'Run the tests?' should be answered with Yes."""
        context = QuestionContext(
            project_path=question_context.project_path,
            question="Run the tests?",
            options=[],
        )
        answer = await brain.answer_question("Run the tests?", [], context)
        assert answer.selected == "Yes"
        assert answer.confidence >= 0.75

    @pytest.mark.asyncio
    async def test_execute_build(self, brain: RuleBasedBrain, question_context: QuestionContext):
        """'Execute build?' should be answered with Yes."""
        context = QuestionContext(
            project_path=question_context.project_path,
            question="Execute the build?",
            options=[],
        )
        answer = await brain.answer_question("Execute the build?", [], context)
        assert answer.selected == "Yes"
        assert answer.confidence >= 0.75

    @pytest.mark.asyncio
    async def test_install_dependencies(
        self, brain: RuleBasedBrain, question_context: QuestionContext
    ):
        """'Install dependencies?' should be answered with Yes."""
        context = QuestionContext(
            project_path=question_context.project_path,
            question="Install the dependencies?",
            options=[],
        )
        answer = await brain.answer_question("Install the dependencies?", [], context)
        assert answer.selected == "Yes"
        assert answer.confidence >= 0.7

    @pytest.mark.asyncio
    async def test_create_file(self, brain: RuleBasedBrain, question_context: QuestionContext):
        """'Create a new file?' should be answered with Yes."""
        context = QuestionContext(
            project_path=question_context.project_path,
            question="Create a new file?",
            options=[],
        )
        answer = await brain.answer_question("Create a new file?", [], context)
        assert answer.selected == "Yes"
        assert answer.confidence >= 0.7

    @pytest.mark.asyncio
    async def test_update_code(self, brain: RuleBasedBrain, question_context: QuestionContext):
        """'Update the code?' should be answered with Yes."""
        context = QuestionContext(
            project_path=question_context.project_path,
            question="Update the code?",
            options=[],
        )
        answer = await brain.answer_question("Update the code?", [], context)
        assert answer.selected == "Yes"
        assert answer.confidence >= 0.7


class TestAnswerQuestionGit:
    """Tests for answering Git-related questions."""

    @pytest.mark.asyncio
    async def test_commit_changes(self, brain: RuleBasedBrain, question_context: QuestionContext):
        """'Commit the changes?' should be answered with Yes."""
        context = QuestionContext(
            project_path=question_context.project_path,
            question="Commit the changes?",
            options=[],
        )
        answer = await brain.answer_question("Commit the changes?", [], context)
        assert answer.selected == "Yes"
        assert answer.confidence >= 0.65

    @pytest.mark.asyncio
    async def test_push_to_remote(self, brain: RuleBasedBrain, question_context: QuestionContext):
        """'Push to remote?' should be answered with Yes but lower confidence."""
        context = QuestionContext(
            project_path=question_context.project_path,
            question="Push to origin?",
            options=[],
        )
        answer = await brain.answer_question("Push to origin?", [], context)
        assert answer.selected == "Yes"
        assert answer.confidence >= 0.6
        assert answer.confidence < 0.8  # Lower confidence for push


class TestAnswerQuestionTypeSelection:
    """Tests for answering type/option selection questions."""

    @pytest.mark.asyncio
    async def test_which_type_selects_first(
        self, brain: RuleBasedBrain, question_context: QuestionContext
    ):
        """'Which type?' should select first option."""
        options = [{"label": "TypeA"}, {"label": "TypeB"}]
        context = QuestionContext(
            project_path=question_context.project_path,
            question="Which type should I use?",
            options=options,
        )
        answer = await brain.answer_question("Which type should I use?", options, context)
        assert answer.selected == "TypeA"
        assert answer.confidence >= 0.5

    @pytest.mark.asyncio
    async def test_what_format_selects_first(
        self, brain: RuleBasedBrain, question_context: QuestionContext
    ):
        """'What format?' should select first option."""
        options = [{"label": "JSON"}, {"label": "YAML"}]
        context = QuestionContext(
            project_path=question_context.project_path,
            question="What format should I use?",
            options=options,
        )
        answer = await brain.answer_question("What format should I use?", options, context)
        assert answer.selected == "JSON"


class TestAnswerQuestionUnknown:
    """Tests for unknown questions."""

    @pytest.mark.asyncio
    async def test_unknown_question_low_confidence(
        self, brain: RuleBasedBrain, question_context: QuestionContext
    ):
        """Unknown questions should return low confidence."""
        context = QuestionContext(
            project_path=question_context.project_path,
            question="What is the meaning of life?",
            options=[],
        )
        answer = await brain.answer_question("What is the meaning of life?", [], context)
        assert answer.confidence == 0.0
        assert answer.selected is None

    @pytest.mark.asyncio
    async def test_complex_question_escalates(
        self, brain: RuleBasedBrain, question_context: QuestionContext
    ):
        """Complex questions should be escalated."""
        context = QuestionContext(
            project_path=question_context.project_path,
            question="How should I architect the authentication system?",
            options=[],
        )
        answer = await brain.answer_question(
            "How should I architect the authentication system?", [], context
        )
        assert answer.confidence == 0.0
        assert "requires human" in answer.reasoning.lower()


class TestEvaluatePermissionGit:
    """Tests for evaluating Git permission requests."""

    @pytest.mark.asyncio
    async def test_git_status_approved(
        self, brain: RuleBasedBrain, permission_context: PermissionContext
    ):
        """git status should be approved."""
        decision = await brain.evaluate_permission(
            "Bash", {"command": "git status"}, permission_context
        )
        assert decision.approve is True
        assert decision.confidence >= 0.8

    @pytest.mark.asyncio
    async def test_git_log_approved(
        self, brain: RuleBasedBrain, permission_context: PermissionContext
    ):
        """git log should be approved."""
        decision = await brain.evaluate_permission(
            "Bash", {"command": "git log --oneline"}, permission_context
        )
        assert decision.approve is True

    @pytest.mark.asyncio
    async def test_git_diff_approved(
        self, brain: RuleBasedBrain, permission_context: PermissionContext
    ):
        """git diff should be approved."""
        decision = await brain.evaluate_permission(
            "Bash", {"command": "git diff HEAD"}, permission_context
        )
        assert decision.approve is True

    @pytest.mark.asyncio
    async def test_git_commit_approved(
        self, brain: RuleBasedBrain, permission_context: PermissionContext
    ):
        """git commit should be approved."""
        decision = await brain.evaluate_permission(
            "Bash", {"command": "git commit -m 'fix bug'"}, permission_context
        )
        assert decision.approve is True
        assert decision.confidence >= 0.7

    @pytest.mark.asyncio
    async def test_git_push_approved(
        self, brain: RuleBasedBrain, permission_context: PermissionContext
    ):
        """git push should be approved."""
        decision = await brain.evaluate_permission(
            "Bash", {"command": "git push origin main"}, permission_context
        )
        assert decision.approve is True
        assert decision.confidence >= 0.65

    @pytest.mark.asyncio
    async def test_git_push_force_denied(
        self, brain: RuleBasedBrain, permission_context: PermissionContext
    ):
        """git push --force should be denied."""
        decision = await brain.evaluate_permission(
            "Bash", {"command": "git push --force origin main"}, permission_context
        )
        assert decision.approve is False
        assert decision.confidence >= 0.85


class TestEvaluatePermissionNetwork:
    """Tests for evaluating network operation permissions."""

    @pytest.mark.asyncio
    async def test_curl_download_approved(
        self, brain: RuleBasedBrain, permission_context: PermissionContext
    ):
        """curl with -o should be approved."""
        decision = await brain.evaluate_permission(
            "Bash", {"command": "curl -o file.txt https://example.com/file"}, permission_context
        )
        assert decision.approve is True

    @pytest.mark.asyncio
    async def test_curl_general_approved_cautiously(
        self, brain: RuleBasedBrain, permission_context: PermissionContext
    ):
        """curl general use should be approved with caution."""
        decision = await brain.evaluate_permission(
            "Bash", {"command": "curl https://api.example.com"}, permission_context
        )
        assert decision.approve is True
        assert decision.confidence >= 0.55
        assert decision.confidence < 0.75  # Lower confidence

    @pytest.mark.asyncio
    async def test_wget_approved_cautiously(
        self, brain: RuleBasedBrain, permission_context: PermissionContext
    ):
        """wget should be approved with caution."""
        decision = await brain.evaluate_permission(
            "Bash", {"command": "wget https://example.com/file.tar.gz"}, permission_context
        )
        assert decision.approve is True
        assert decision.confidence >= 0.55


class TestEvaluatePermissionDocker:
    """Tests for evaluating Docker permission requests."""

    @pytest.mark.asyncio
    async def test_docker_build_approved(
        self, brain: RuleBasedBrain, permission_context: PermissionContext
    ):
        """docker build should be approved."""
        decision = await brain.evaluate_permission(
            "Bash", {"command": "docker build -t myapp ."}, permission_context
        )
        assert decision.approve is True
        assert decision.confidence >= 0.65

    @pytest.mark.asyncio
    async def test_docker_run_approved(
        self, brain: RuleBasedBrain, permission_context: PermissionContext
    ):
        """docker run should be approved."""
        decision = await brain.evaluate_permission(
            "Bash", {"command": "docker run -d myapp"}, permission_context
        )
        assert decision.approve is True

    @pytest.mark.asyncio
    async def test_docker_ps_approved(
        self, brain: RuleBasedBrain, permission_context: PermissionContext
    ):
        """docker ps should be approved."""
        decision = await brain.evaluate_permission(
            "Bash", {"command": "docker ps"}, permission_context
        )
        assert decision.approve is True

    @pytest.mark.asyncio
    async def test_docker_rm_denied(
        self, brain: RuleBasedBrain, permission_context: PermissionContext
    ):
        """docker rm should be denied."""
        decision = await brain.evaluate_permission(
            "Bash", {"command": "docker rm container_id"}, permission_context
        )
        assert decision.approve is False

    @pytest.mark.asyncio
    async def test_docker_prune_denied(
        self, brain: RuleBasedBrain, permission_context: PermissionContext
    ):
        """docker system prune should be denied."""
        decision = await brain.evaluate_permission(
            "Bash", {"command": "docker system prune -a"}, permission_context
        )
        assert decision.approve is False


class TestEvaluatePermissionConfigFiles:
    """Tests for evaluating config file permission requests."""

    @pytest.mark.asyncio
    async def test_write_package_json_approved(
        self, brain: RuleBasedBrain, permission_context: PermissionContext
    ):
        """Writing package.json should be approved."""
        decision = await brain.evaluate_permission(
            "Write", {"file_path": "/project/package.json"}, permission_context
        )
        assert decision.approve is True
        assert decision.confidence >= 0.65

    @pytest.mark.asyncio
    async def test_edit_package_json_approved(
        self, brain: RuleBasedBrain, permission_context: PermissionContext
    ):
        """Editing package.json should be approved."""
        decision = await brain.evaluate_permission(
            "Edit",
            {"file_path": "/project/package.json", "old_string": "a", "new_string": "b"},
            permission_context,
        )
        assert decision.approve is True

    @pytest.mark.asyncio
    async def test_write_tsconfig_approved(
        self, brain: RuleBasedBrain, permission_context: PermissionContext
    ):
        """Writing tsconfig.json should be approved."""
        decision = await brain.evaluate_permission(
            "Write", {"file_path": "/project/tsconfig.json"}, permission_context
        )
        assert decision.approve is True

    @pytest.mark.asyncio
    async def test_write_pyproject_approved(
        self, brain: RuleBasedBrain, permission_context: PermissionContext
    ):
        """Writing pyproject.toml should be approved."""
        decision = await brain.evaluate_permission(
            "Write", {"file_path": "/project/pyproject.toml"}, permission_context
        )
        assert decision.approve is True


class TestEvaluatePermissionFileOps:
    """Tests for evaluating file operation permissions."""

    @pytest.mark.asyncio
    async def test_simple_rm_approved(
        self, brain: RuleBasedBrain, permission_context: PermissionContext
    ):
        """Simple rm (not -rf) should be approved."""
        decision = await brain.evaluate_permission(
            "Bash", {"command": "rm temp.txt"}, permission_context
        )
        assert decision.approve is True

    @pytest.mark.asyncio
    async def test_mv_approved(self, brain: RuleBasedBrain, permission_context: PermissionContext):
        """mv should be approved."""
        decision = await brain.evaluate_permission(
            "Bash", {"command": "mv old.py new.py"}, permission_context
        )
        assert decision.approve is True


class TestEvaluatePermissionTask:
    """Tests for evaluating Task tool permissions."""

    @pytest.mark.asyncio
    async def test_task_approved(
        self, brain: RuleBasedBrain, permission_context: PermissionContext
    ):
        """Task tool should be approved."""
        decision = await brain.evaluate_permission(
            "Task", {"prompt": "Search for files", "subagent_type": "Explore"}, permission_context
        )
        assert decision.approve is True
        assert decision.confidence >= 0.7


class TestEvaluatePermissionUnknown:
    """Tests for evaluating unknown permission requests."""

    @pytest.mark.asyncio
    async def test_unknown_tool_escalates(
        self, brain: RuleBasedBrain, permission_context: PermissionContext
    ):
        """Unknown tools should be escalated."""
        decision = await brain.evaluate_permission(
            "UnknownTool", {"foo": "bar"}, permission_context
        )
        assert decision.approve is False
        assert decision.requires_human is True
        assert decision.confidence == 0.0

    @pytest.mark.asyncio
    async def test_unknown_bash_command_escalates(
        self, brain: RuleBasedBrain, permission_context: PermissionContext
    ):
        """Unknown bash commands should be escalated."""
        decision = await brain.evaluate_permission(
            "Bash", {"command": "some_obscure_command --flag"}, permission_context
        )
        assert decision.approve is False
        assert decision.requires_human is True


class TestConfidenceThreshold:
    """Tests for confidence threshold behavior."""

    def test_default_threshold(self, brain: RuleBasedBrain):
        """Default threshold should be 0.7."""
        assert brain.get_confidence_threshold() == 0.7

    def test_custom_threshold(self):
        """Custom threshold should be respected."""
        config = RuleBasedConfig(
            question_patterns=default_question_patterns(),
            permission_patterns=default_permission_patterns(),
            confidence_threshold=0.9,
        )
        brain = RuleBasedBrain(config=config)
        assert brain.get_confidence_threshold() == 0.9


class TestCustomPatterns:
    """Tests for custom pattern configuration."""

    @pytest.mark.asyncio
    async def test_custom_question_pattern(self, question_context: QuestionContext):
        """Custom question patterns should work."""
        custom_patterns = (
            QuestionPattern(
                pattern=re.compile(r"deploy to prod", re.IGNORECASE),
                answer="No",
                confidence=0.95,
                reasoning="Never auto-deploy to prod",
            ),
        )
        config = RuleBasedConfig(question_patterns=custom_patterns)
        brain = RuleBasedBrain(config=config)

        context = QuestionContext(
            project_path=question_context.project_path,
            question="Deploy to prod?",
            options=[],
        )
        answer = await brain.answer_question("Deploy to prod?", [], context)
        assert answer.selected == "No"
        assert answer.confidence == 0.95

    @pytest.mark.asyncio
    async def test_custom_permission_pattern(self, permission_context: PermissionContext):
        """Custom permission patterns should work."""
        custom_patterns = (
            PermissionPattern(
                tool_name="Bash",
                input_pattern=re.compile(r"npm publish"),
                approve=False,
                confidence=0.99,
                reasoning="Never auto-publish",
            ),
        )
        config = RuleBasedConfig(permission_patterns=custom_patterns)
        brain = RuleBasedBrain(config=config)

        decision = await brain.evaluate_permission(
            "Bash", {"command": "npm publish"}, permission_context
        )
        assert decision.approve is False
        assert decision.confidence == 0.99


class TestInputStringExtraction:
    """Tests for input string extraction from tool input."""

    def test_bash_command_extraction(self, brain: RuleBasedBrain):
        """Bash command should be extracted correctly."""
        result = brain._get_input_string("Bash", {"command": "npm install"})
        assert result == "npm install"

    def test_file_path_extraction(self, brain: RuleBasedBrain):
        """File path should be extracted correctly."""
        result = brain._get_input_string("Write", {"file_path": "/path/to/file.py"})
        assert result == "/path/to/file.py"

    def test_glob_pattern_extraction(self, brain: RuleBasedBrain):
        """Glob pattern should be extracted correctly."""
        result = brain._get_input_string("Glob", {"pattern": "**/*.py"})
        assert result == "**/*.py"

    def test_task_prompt_extraction(self, brain: RuleBasedBrain):
        """Task prompt should be extracted correctly."""
        result = brain._get_input_string("Task", {"prompt": "Search for files"})
        assert result == "Search for files"

    def test_unknown_tool_fallback(self, brain: RuleBasedBrain):
        """Unknown tools should use string conversion."""
        result = brain._get_input_string("CustomTool", {"key": "value"})
        assert "key" in result
        assert "value" in result


class TestIntegrationWithDecisionEngine:
    """Integration tests with DecisionEngine."""

    @pytest.mark.asyncio
    async def test_brain_with_decision_engine(self, temp_project: Path):
        """Brain should integrate with DecisionEngine."""
        from perpetualcc.brain.base import PermissionContext
        from perpetualcc.core.decision_engine import DecisionEngine

        brain = RuleBasedBrain()
        engine = DecisionEngine(temp_project, brain=brain)
        context = PermissionContext(project_path=str(temp_project))

        # Medium-risk git command should be evaluated by brain
        decision = await engine.decide_permission_async(
            "Bash", {"command": "git commit -m 'test'"}, context=context
        )
        # Brain should approve git commit
        assert decision.approve is True

    @pytest.mark.asyncio
    async def test_brain_denies_in_decision_engine(self, temp_project: Path):
        """Brain denial should flow through DecisionEngine."""
        from perpetualcc.brain.base import PermissionContext
        from perpetualcc.core.decision_engine import DecisionEngine

        brain = RuleBasedBrain()
        engine = DecisionEngine(temp_project, brain=brain)
        context = PermissionContext(project_path=str(temp_project))

        # git push --force should be denied by brain
        decision = await engine.decide_permission_async(
            "Bash", {"command": "git push --force origin main"}, context=context
        )
        assert decision.approve is False
