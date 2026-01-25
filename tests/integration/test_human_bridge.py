"""Integration tests for the Human Bridge system.

These tests verify end-to-end functionality:
1. HumanBridge + EscalationQueue + Notifier integration
2. MasterAgent escalation flow
3. CLI command integration
4. Real-world user scenarios
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from perpetualcc.brain.rule_based import RuleBasedBrain
from perpetualcc.claude.types import AskQuestionEvent, Question, QuestionOption, ToolUseEvent
from perpetualcc.core.decision_engine import DecisionEngine
from perpetualcc.core.master_agent import ActionType, MasterAgent, MasterAgentConfig
from perpetualcc.core.session_manager import (
    ManagedSession,
    SessionManager,
    SessionManagerConfig,
)
from perpetualcc.human.escalation import (
    EscalationQueue,
    EscalationRequest,
    EscalationStatus,
    EscalationType,
    HumanBridge,
)


@pytest.fixture
def temp_dir() -> Path:
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def session(temp_dir: Path) -> ManagedSession:
    """Create a test session."""
    return ManagedSession.create(
        project_path=temp_dir,
        task="Implement user authentication",
    )


class TestHumanBridgeIntegration:
    """Integration tests for HumanBridge."""

    @pytest.mark.asyncio
    async def test_full_escalation_flow(self, temp_dir: Path, session: ManagedSession):
        """Test complete escalation flow: create -> wait -> respond."""
        bridge = HumanBridge(data_dir=temp_dir, enable_notifications=False)
        await bridge.initialize()

        try:
            # Start escalation in background
            async def escalate_and_wait():
                return await bridge.escalate_question(
                    session=session,
                    question="Which database?",
                    options=["PostgreSQL", "SQLite"],
                    suggestion="PostgreSQL",
                    confidence=0.6,
                )

            # Start waiting
            wait_task = asyncio.create_task(escalate_and_wait())

            # Give time for escalation to be created
            await asyncio.sleep(0.1)

            # Get pending and respond
            pending = await bridge.get_pending()
            assert len(pending) == 1

            await bridge.respond(pending[0].id, "SQLite")

            # Wait for result
            response = await wait_task
            assert response == "SQLite"

        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_permission_escalation_approve(
        self, temp_dir: Path, session: ManagedSession
    ):
        """Test permission escalation with approval."""
        bridge = HumanBridge(data_dir=temp_dir, enable_notifications=False)
        await bridge.initialize()

        try:
            async def escalate():
                return await bridge.escalate_permission(
                    session=session,
                    tool_name="Bash",
                    tool_input={"command": "rm -rf node_modules"},
                    risk_level="HIGH",
                    suggestion="Deny",
                    confidence=0.3,
                )

            wait_task = asyncio.create_task(escalate())
            await asyncio.sleep(0.1)

            pending = await bridge.get_pending()
            assert len(pending) == 1
            assert pending[0].escalation_type == EscalationType.PERMISSION

            await bridge.respond(pending[0].id, "Approve")

            response = await wait_task
            assert response == "approve"

        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_permission_escalation_deny(
        self, temp_dir: Path, session: ManagedSession
    ):
        """Test permission escalation with denial."""
        bridge = HumanBridge(data_dir=temp_dir, enable_notifications=False)
        await bridge.initialize()

        try:
            async def escalate():
                return await bridge.escalate_permission(
                    session=session,
                    tool_name="Bash",
                    tool_input={"command": "sudo rm -rf /"},
                    risk_level="HIGH",
                )

            wait_task = asyncio.create_task(escalate())
            await asyncio.sleep(0.1)

            pending = await bridge.get_pending()
            await bridge.respond(pending[0].id, "Deny")

            response = await wait_task
            assert response == "deny"

        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_error_escalation_flow(
        self, temp_dir: Path, session: ManagedSession
    ):
        """Test error escalation flow."""
        bridge = HumanBridge(data_dir=temp_dir, enable_notifications=False)
        await bridge.initialize()

        try:
            async def escalate():
                return await bridge.escalate_error(
                    session=session,
                    error_message="Build failed: Cannot find module 'lodash'",
                    options=["Retry", "Skip", "Abort"],
                )

            wait_task = asyncio.create_task(escalate())
            await asyncio.sleep(0.1)

            pending = await bridge.get_pending()
            assert len(pending) == 1
            assert pending[0].escalation_type == EscalationType.ERROR

            await bridge.respond(pending[0].id, "Retry")

            response = await wait_task
            assert response == "Retry"

        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_non_waiting_escalation(
        self, temp_dir: Path, session: ManagedSession
    ):
        """Test escalation without waiting."""
        bridge = HumanBridge(data_dir=temp_dir, enable_notifications=False)
        await bridge.initialize()

        try:
            # Escalate without waiting
            response = await bridge.escalate_question(
                session=session,
                question="Should I proceed?",
                wait_for_response=False,
            )

            assert response is None

            # But escalation should be pending
            pending = await bridge.get_pending()
            assert len(pending) == 1

        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_statistics(self, temp_dir: Path, session: ManagedSession):
        """Test escalation statistics."""
        bridge = HumanBridge(data_dir=temp_dir, enable_notifications=False)
        await bridge.initialize()

        try:
            # Create some escalations
            for i in range(3):
                await bridge.escalate_question(
                    session=session,
                    question=f"Question {i}?",
                    wait_for_response=False,
                )

            # Respond to one
            pending = await bridge.get_pending()
            await bridge.respond(pending[0].id, "Yes")

            stats = await bridge.get_statistics()
            assert stats["total_escalations"] == 3
            assert stats["pending_count"] == 2

        finally:
            await bridge.close()


class TestMasterAgentWithHumanBridge:
    """Integration tests for MasterAgent + HumanBridge."""

    @pytest.fixture
    def session_manager(self, temp_dir: Path) -> SessionManager:
        """Create session manager."""
        config = SessionManagerConfig(data_dir=temp_dir)
        return SessionManager(config)

    @pytest.fixture
    def decision_engine(self, temp_dir: Path) -> DecisionEngine:
        """Create decision engine."""
        brain = RuleBasedBrain()
        return DecisionEngine(project_path=temp_dir, brain=brain)

    @pytest.mark.asyncio
    async def test_master_agent_escalates_complex_question(
        self,
        temp_dir: Path,
        session_manager: SessionManager,
        decision_engine: DecisionEngine,
        session: ManagedSession,
    ):
        """Test MasterAgent escalates questions with low confidence."""
        bridge = HumanBridge(data_dir=temp_dir, enable_notifications=False)
        await bridge.initialize()

        agent = MasterAgent(
            session_manager=session_manager,
            decision_engine=decision_engine,
            brain=RuleBasedBrain(),
            human_bridge=bridge,
        )

        try:
            # Complex question that won't match patterns
            event = AskQuestionEvent(
                tool_use_id="q1",
                questions=[
                    Question(
                        question="Which microservices architecture should I use?",
                        options=[
                            QuestionOption(label="Event Sourcing"),
                            QuestionOption(label="REST + Saga"),
                        ],
                    )
                ],
            )

            analysis = await agent._think(event, session)
            action = await agent._decide(analysis, session)

            # Should escalate to human
            assert action.type == ActionType.ESCALATE_TO_HUMAN
            assert action.requires_human is True

        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_master_agent_executes_human_escalation(
        self,
        temp_dir: Path,
        session_manager: SessionManager,
        decision_engine: DecisionEngine,
        session: ManagedSession,
    ):
        """Test MasterAgent._execute calls HumanBridge for escalation."""
        bridge = HumanBridge(data_dir=temp_dir, enable_notifications=False)
        await bridge.initialize()

        agent = MasterAgent(
            session_manager=session_manager,
            decision_engine=decision_engine,
            human_bridge=bridge,
        )

        try:
            # Create escalation action
            from perpetualcc.core.master_agent import Action

            action = Action(
                type=ActionType.ESCALATE_TO_HUMAN,
                reason="Low confidence",
                requires_human=True,
                metadata={
                    "question": "Which approach?",
                    "options": [{"label": "A"}, {"label": "B"}],
                },
            )

            # Execute in background
            async def execute_and_respond():
                result_task = asyncio.create_task(
                    agent._execute(action, session)
                )
                await asyncio.sleep(0.1)

                pending = await bridge.get_pending()
                if pending:
                    await bridge.respond(pending[0].id, "A")

                return await result_task

            result = await execute_and_respond()

            assert result["action"] == "escalated_to_human"
            assert result.get("human_response") == "A"

        finally:
            await bridge.close()


class TestCLIEscalationCommands:
    """Integration tests for CLI escalation commands."""

    @pytest.mark.asyncio
    async def test_pending_command_shows_escalations(self, temp_dir: Path):
        """Test pcc pending command lists escalations."""
        queue = EscalationQueue(data_dir=temp_dir)
        await queue.initialize()

        try:
            # Create some escalations
            await queue.escalate(
                session_id="session-123",
                escalation_type=EscalationType.QUESTION,
                context="",
                question="Question 1?",
            )
            await queue.escalate(
                session_id="session-456",
                escalation_type=EscalationType.PERMISSION,
                context="",
                question="Allow X?",
            )

            pending = await queue.get_pending()
            assert len(pending) == 2

        finally:
            await queue.close()

    @pytest.mark.asyncio
    async def test_respond_by_index(self, temp_dir: Path):
        """Test responding to escalation by index."""
        queue = EscalationQueue(data_dir=temp_dir)
        await queue.initialize()

        try:
            # Create escalations
            r1 = await queue.escalate(
                session_id="session-123",
                escalation_type=EscalationType.QUESTION,
                context="",
                question="Q1?",
            )
            r2 = await queue.escalate(
                session_id="session-456",
                escalation_type=EscalationType.QUESTION,
                context="",
                question="Q2?",
            )

            # Get pending (ordered by creation)
            pending = await queue.get_pending()

            # Respond to first by ID
            await queue.respond(pending[0].id, "Answer 1")

            # Check response
            updated = await queue.get_request(r1.id)
            assert updated.response == "Answer 1"

        finally:
            await queue.close()


class TestRealWorldScenarios:
    """End-to-end tests for real-world usage scenarios."""

    @pytest.mark.asyncio
    async def test_scenario_developer_reviews_dangerous_command(self, temp_dir: Path):
        """Scenario: Developer reviews and approves a dangerous command."""
        session = ManagedSession.create(
            project_path=temp_dir,
            task="Clean up and reinstall dependencies",
        )

        bridge = HumanBridge(data_dir=temp_dir, enable_notifications=False)
        await bridge.initialize()

        try:
            # Claude wants to run cleanup
            async def claude_requests_permission():
                return await bridge.escalate_permission(
                    session=session,
                    tool_name="Bash",
                    tool_input={"command": "rm -rf node_modules && npm install"},
                    risk_level="HIGH",
                    suggestion="Deny",
                    confidence=0.2,
                )

            # Start request
            request_task = asyncio.create_task(claude_requests_permission())
            await asyncio.sleep(0.1)

            # Developer sees pending escalation
            pending = await bridge.get_pending()
            assert len(pending) == 1
            assert "rm -rf" in pending[0].question

            # Developer approves after reviewing
            await bridge.respond(pending[0].id, "Approve")

            # Claude gets approval
            result = await request_task
            assert result == "approve"

        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_scenario_multiple_sessions_with_escalations(self, temp_dir: Path):
        """Scenario: Multiple concurrent sessions with escalations."""
        bridge = HumanBridge(data_dir=temp_dir, enable_notifications=False)
        await bridge.initialize()

        sessions = [
            ManagedSession.create(project_path=temp_dir, task=f"Task {i}")
            for i in range(3)
        ]

        try:
            # Each session creates an escalation
            tasks = []
            for i, session in enumerate(sessions):
                async def create_escalation(s=session, idx=i):
                    return await bridge.escalate_question(
                        session=s,
                        question=f"Session {idx} question?",
                        wait_for_response=False,
                    )

                tasks.append(asyncio.create_task(create_escalation()))

            await asyncio.gather(*tasks)

            # All pending
            all_pending = await bridge.get_pending()
            assert len(all_pending) == 3

            # Filter by session
            s0_pending = await bridge.get_pending(session_id=sessions[0].id)
            assert len(s0_pending) == 1

            # Respond to one
            await bridge.respond(all_pending[0].id, "Yes")

            # Two remaining
            remaining = await bridge.get_pending()
            assert len(remaining) == 2

        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_scenario_session_cleanup_cancels_escalations(self, temp_dir: Path):
        """Scenario: Stopping a session cancels its pending escalations."""
        session = ManagedSession.create(
            project_path=temp_dir,
            task="Test task",
        )

        queue = EscalationQueue(data_dir=temp_dir)
        await queue.initialize()

        try:
            # Create multiple escalations for session
            for i in range(3):
                await queue.escalate(
                    session_id=session.id,
                    escalation_type=EscalationType.QUESTION,
                    context="",
                    question=f"Q{i}?",
                )

            assert len(await queue.get_pending(session_id=session.id)) == 3

            # Clear session escalations (simulating session stop)
            count = await queue.clear_session(session.id)
            assert count == 3

            # No more pending
            assert len(await queue.get_pending(session_id=session.id)) == 0

        finally:
            await queue.close()
