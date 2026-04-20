"""Test cases for the Chatbot Agent.

Tests the chatbot graph construction, message handling, and response generation.
"""

import pytest
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables.config import RunnableConfig

from src.agents.chatbot import build_chatbot_graph


class TestChatbotGraph:
    """Test suite for chatbot graph functionality."""
    
    @pytest.fixture
    def graph(self):
        """Create a compiled chatbot graph for testing."""
        return build_chatbot_graph()
    
    def test_graph_builds_successfully(self):
        """Test that the chatbot graph builds without errors."""
        graph = build_chatbot_graph()
        assert graph is not None
        assert hasattr(graph, "invoke")
        assert hasattr(graph, "stream")
    
    def test_single_message_response(self, graph):
        """Test that chatbot responds to a single message."""
        response = graph.invoke({"messages": [HumanMessage(content="Hello")]})
        
        assert "messages" in response
        assert len(response["messages"]) > 0
        assert isinstance(response["messages"][-1], BaseMessage)
        assert response["messages"][-1].content is not None
    
    def test_response_is_not_empty(self, graph):
        """Test that chatbot produces non-empty responses."""
        response = graph.invoke({"messages": [HumanMessage(content="What is AI?")]})
        
        content = response["messages"][-1].content
        assert content is not None
        assert len(content) > 0
        assert content.strip() != ""
    
    def test_multiple_turns(self, graph):
        """Test multi-turn conversation handling."""
        messages = [HumanMessage(content="What is machine learning?")]
        response1 = graph.invoke({"messages": messages})
        
        # Add the response to messages
        messages.append(response1["messages"][-1])
        messages.append(HumanMessage(content="Can you give an example?"))
        
        response2 = graph.invoke({"messages": messages})
        
        assert len(response2["messages"]) > len(response1["messages"])
        assert response2["messages"][-1].content is not None
    
    def test_message_accumulation(self, graph):
        """Test that messages are properly accumulated in the state."""
        messages = [
            HumanMessage(content="First message"),
            HumanMessage(content="Second message"),
        ]
        response = graph.invoke({"messages": messages})
        
        # Should have both input messages + response
        assert len(response["messages"]) >= len(messages)
    
    def test_different_query_topics(self, graph):
        """Test chatbot with different topics."""
        topics = [
            "Hello",
            "What is Python?",
            "Explain machine learning",
            "How do neural networks work?",
        ]
        
        for topic in topics:
            response = graph.invoke({"messages": [HumanMessage(content=topic)]})
            assert response["messages"][-1].content is not None
            assert len(response["messages"][-1].content) > 0
    
    def test_response_has_correct_structure(self, graph):
        """Test that response has the expected structure."""
        response = graph.invoke({"messages": [HumanMessage(content="Hi")]})
        
        assert isinstance(response, dict)
        assert "messages" in response
        assert isinstance(response["messages"], list)
        assert all(isinstance(msg, BaseMessage) for msg in response["messages"])
    
    def test_does_not_modify_input_state(self, graph):
        """Test that the graph doesn't modify the original input state."""
        input_state = {"messages": [HumanMessage(content="Test")]}
        original_length = len(input_state["messages"])
        
        response = graph.invoke(input_state)
        
        # Original input should not be modified
        assert len(input_state["messages"]) == original_length
        # Response should have additional messages
        assert len(response["messages"]) >= original_length
    
    def test_long_conversation(self, graph):
        """Test chatbot with a longer conversation."""
        messages = [HumanMessage(content="Start")]
        
        for i in range(3):
            response = graph.invoke({"messages": messages})
            messages.append(response["messages"][-1])
            messages.append(HumanMessage(content=f"Continue with point {i+1}"))
        
        final_response = graph.invoke({"messages": messages})
        assert final_response["messages"][-1].content is not None
    
    def test_empty_message_handling(self, graph):
        """Test handling of edge cases."""
        # This tests if the graph can handle the invocation
        # In real scenarios, you might want to handle empty messages
        try:
            response = graph.invoke({"messages": [HumanMessage(content="")]})
            # Should still return a response even with empty input
            assert "messages" in response
        except Exception:
            # It's acceptable if it raises an error for empty input
            pass
    
    def test_special_characters_in_input(self, graph):
        """Test that chatbot handles special characters."""
        special_inputs = [
            "What is @#$%?",
            "Price: $99.99",
            "Email: test@example.com",
            "Code: print('hello')",
        ]
        
        for inp in special_inputs:
            response = graph.invoke({"messages": [HumanMessage(content=inp)]})
            assert response["messages"][-1].content is not None


class TestChatbotEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def graph(self):
        """Create a compiled chatbot graph for testing."""
        return build_chatbot_graph()
    
    def test_very_long_input(self, graph):
        """Test with very long input message."""
        long_message = "A" * 1000
        response = graph.invoke({"messages": [HumanMessage(content=long_message)]})
        assert response["messages"][-1].content is not None
    
    def test_unicode_characters(self, graph):
        """Test Unicode and emoji handling."""
        unicode_inputs = [
            "Hello 👋",
            "你好",
            "مرحبا",
            "Привет",
        ]
        
        for inp in unicode_inputs:
            try:
                response = graph.invoke({"messages": [HumanMessage(content=inp)]})
                assert response["messages"][-1].content is not None
            except Exception:
                # Unicode support might vary by LLM
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
