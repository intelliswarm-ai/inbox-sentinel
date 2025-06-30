"""
LangChain-based orchestrator for email analysis using MCP servers as tools
"""

import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

try:
    from langchain.tools import Tool
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import PromptTemplate
    from langchain.schema import BaseMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Define dummy classes to prevent errors
    Tool = None
    AgentExecutor = None
    PromptTemplate = None

# Try to import Ollama (try new package first, fall back to old)
try:
    from langchain_ollama import OllamaLLM as Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.llms import Ollama
        OLLAMA_AVAILABLE = True
    except ImportError:
        Ollama = None
        OLLAMA_AVAILABLE = False

# Try to import OpenAI separately (optional)
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    ChatOpenAI = None
    OPENAI_AVAILABLE = False

from inbox_sentinel.core.types import Email, PredictionResult
from inbox_sentinel.ml.models import (
    NaiveBayesDetector,
    SVMDetector,
    RandomForestDetector,
    LogisticRegressionDetector,
    NeuralNetworkDetector
)


@dataclass
class OrchestrationConfig:
    """Configuration for the orchestrator"""
    llm_provider: str = "ollama"  # "ollama" or "openai"
    model_name: str = "llama2"  # for ollama: llama2, mistral, etc.
    temperature: float = 0.1
    max_iterations: int = 10
    verbose: bool = True


class MCPServerTool:
    """Wrapper to convert MCP server into LangChain tool"""
    
    def __init__(self, detector_class, name: str, description: str):
        self.detector_class = detector_class
        self.name = name
        self.description = description
        self.detector = None
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Lazy initialization of detector"""
        if not self._initialized:
            self.detector = self.detector_class()
            await self.detector.initialize(use_pretrained=True)
            self._initialized = True
    
    async def analyze(self, email_json: str) -> str:
        """Analyze email and return results as JSON string"""
        try:
            # Parse email data
            email_data = json.loads(email_json)
            email = Email(
                content=email_data.get('content', ''),
                subject=email_data.get('subject', ''),
                sender=email_data.get('sender', '')
            )
            
            # Ensure detector is initialized
            await self._ensure_initialized()
            
            # Analyze
            result = await self.detector.analyze(email)
            
            # Convert to dict for JSON serialization
            return json.dumps({
                'model_name': result.model_name,
                'algorithm': result.algorithm,
                'is_spam': result.is_spam,
                'spam_probability': result.spam_probability,
                'ham_probability': result.ham_probability,
                'confidence': result.confidence,
                'top_features': result.features[:3] if result.features else [],
                'error': result.error
            })
            
        except Exception as e:
            return json.dumps({
                'error': str(e),
                'model_name': self.name
            })


class LangChainOrchestrator:
    """Orchestrates email analysis using LangChain and MCP servers"""
    
    def __init__(self, config: OrchestrationConfig = None):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. Please install it with:\n"
                "pip install langchain langchain-community langchain-openai"
            )
        self.config = config or OrchestrationConfig()
        self.tools = []
        self.agent = None
        self.executor = None
        self._setup_tools()
        self._setup_llm()
    
    def _setup_tools(self):
        """Set up MCP servers as LangChain tools"""
        # Define MCP server tools
        mcp_tools = [
            (NaiveBayesDetector, "naive_bayes_analyzer", 
             "Naive Bayes spam detector. Good for text classification with 96% accuracy. Fast and interpretable."),
            (SVMDetector, "svm_analyzer",
             "Support Vector Machine spam detector. 95% accuracy with RBF kernel. Good for complex patterns."),
            (RandomForestDetector, "random_forest_analyzer",
             "Random Forest spam detector. 94% accuracy. Provides feature importance rankings."),
            (LogisticRegressionDetector, "logistic_regression_analyzer",
             "Logistic Regression spam detector. 95% accuracy. Linear model, highly interpretable."),
            (NeuralNetworkDetector, "neural_network_analyzer",
             "Neural Network spam detector. 96.6% accuracy. Best for complex patterns but less interpretable.")
        ]
        
        # Create LangChain tools
        for detector_class, name, description in mcp_tools:
            mcp_tool = MCPServerTool(detector_class, name, description)
            
            # Create async wrapper
            async def analyze_wrapper(email_json: str, tool=mcp_tool) -> str:
                return await tool.analyze(email_json)
            
            # Create sync wrapper for LangChain
            def sync_wrapper(email_json: str, tool=mcp_tool) -> str:
                import nest_asyncio
                nest_asyncio.apply()
                
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                return loop.run_until_complete(tool.analyze(email_json))
            
            # Create LangChain tool
            langchain_tool = Tool(
                name=name,
                description=f"{description} Input should be JSON with 'content', 'subject', and 'sender' fields.",
                func=sync_wrapper
            )
            
            self.tools.append(langchain_tool)
    
    def _setup_llm(self):
        """Set up the LLM and agent"""
        # Choose LLM based on config
        if self.config.llm_provider == "ollama":
            if not OLLAMA_AVAILABLE:
                raise ImportError(
                    "Ollama support requires langchain-ollama or langchain-community. Install with:\n"
                    "pip install langchain-ollama\n"
                    "or\n"
                    "pip install langchain-community"
                )
            llm = Ollama(
                model=self.config.model_name,
                temperature=self.config.temperature
            )
        else:
            # OpenAI or other providers
            if not OPENAI_AVAILABLE:
                raise ImportError(
                    "OpenAI support requires langchain-openai. Install with:\n"
                    "pip install langchain-openai"
                )
            llm = ChatOpenAI(
                model_name=self.config.model_name,
                temperature=self.config.temperature
            )
        
        # Create prompt template for ReAct agent
        prompt = PromptTemplate.from_template("""You have access to the following tools:

{tools}

Use the following format EXACTLY:

Question: the input question you must answer
Thought: you should always think about what to do
Action: [tool name exactly as shown above]
Action Input: [valid JSON object]
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT: 
- Action must be EXACTLY one of: [{tool_names}]
- Action Input must be a valid JSON object like: {{"content": "email text", "subject": "subject", "sender": "sender@email.com"}}

Example:
Question: Analyze this email for spam
Thought: I need to use multiple spam detectors to analyze this email
Action: naive_bayes_analyzer
Action Input: {{"content": "Win money now!", "subject": "You won!", "sender": "scam@fake.com"}}
Observation: {{"is_spam": true, "confidence": 0.95}}
Thought: Let me check with another detector
Action: neural_network_analyzer
Action Input: {{"content": "Win money now!", "subject": "You won!", "sender": "scam@fake.com"}}
Observation: {{"is_spam": true, "confidence": 0.98}}
Thought: Both detectors agree this is spam with high confidence
Final Answer: This email is spam/phishing with 95-98% confidence based on multiple detectors.

Begin!

Question: {input}
Thought: {agent_scratchpad}""")
        
        # Create agent
        self.agent = create_react_agent(
            llm=llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create executor
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=self.config.verbose,
            max_iterations=self.config.max_iterations,
            handle_parsing_errors=True
        )
    
    async def analyze_email(self, email: Email) -> Dict[str, Any]:
        """Analyze an email using the LLM orchestrator"""
        # Prepare email data
        email_json = json.dumps({
            'content': email.content,
            'subject': email.subject,
            'sender': email.sender
        })
        
        # Create input for agent
        input_text = f"""Analyze this email for spam/phishing. Use at least 3 different tools.

Here is the email data to pass to each tool:
{email_json}"""
        
        # Run agent
        try:
            result = self.executor.invoke({"input": input_text})
            
            # Parse the output to extract structured results
            return {
                'success': True,
                'analysis': result.get('output', ''),
                'intermediate_steps': result.get('intermediate_steps', []),
                'email': {
                    'subject': email.subject,
                    'sender': email.sender
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'email': {
                    'subject': email.subject,
                    'sender': email.sender
                }
            }
    
    def analyze_email_sync(self, email: Email) -> Dict[str, Any]:
        """Synchronous wrapper for analyze_email"""
        try:
            # Check if there's already an event loop running
            loop = asyncio.get_running_loop()
            # If we're here, there's already a loop - create a task
            import concurrent.futures
            import threading
            
            result = None
            exception = None
            
            def run_in_thread():
                nonlocal result, exception
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    result = new_loop.run_until_complete(self.analyze_email(email))
                    new_loop.close()
                except Exception as e:
                    exception = e
            
            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()
            
            if exception:
                raise exception
            return result
            
        except RuntimeError:
            # No event loop running, we can create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.analyze_email(email))
            loop.close()
            return result


class SimpleOrchestrator:
    """Simpler orchestrator without LLM dependency for testing"""
    
    def __init__(self):
        self.detectors = {
            'naive_bayes': NaiveBayesDetector(),
            'svm': SVMDetector(),
            'random_forest': RandomForestDetector(),
            'logistic_regression': LogisticRegressionDetector(),
            'neural_network': NeuralNetworkDetector()
        }
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Initialize all detectors"""
        if not self._initialized:
            for name, detector in self.detectors.items():
                try:
                    await detector.initialize(use_pretrained=True)
                    print(f"✅ Initialized {name}")
                except Exception as e:
                    print(f"❌ Failed to initialize {name}: {e}")
            self._initialized = True
    
    async def analyze_email(self, email: Email) -> Dict[str, Any]:
        """Analyze email with all detectors and provide consensus"""
        await self._ensure_initialized()
        
        results = {}
        spam_votes = 0
        total_confidence = 0
        all_features = []
        
        # Run all detectors
        for name, detector in self.detectors.items():
            if detector.is_trained:
                try:
                    result = await detector.analyze(email)
                    results[name] = {
                        'is_spam': result.is_spam,
                        'confidence': result.confidence,
                        'spam_probability': result.spam_probability,
                        'features': result.features[:3] if result.features else []
                    }
                    
                    if result.is_spam:
                        spam_votes += 1
                    total_confidence += result.confidence
                    
                    if result.features:
                        all_features.extend(result.features[:2])
                        
                except Exception as e:
                    results[name] = {'error': str(e)}
        
        # Calculate consensus
        total_models = len([r for r in results.values() if 'error' not in r])
        consensus_spam = spam_votes > total_models / 2
        avg_confidence = total_confidence / total_models if total_models > 0 else 0
        
        # Create analysis text
        analysis = f"""
Email Analysis Report
====================

Subject: {email.subject}
Sender: {email.sender}

Individual Model Results:
"""
        
        for model, result in results.items():
            if 'error' not in result:
                verdict = "SPAM" if result['is_spam'] else "LEGITIMATE"
                analysis += f"\n- {model}: {verdict} (confidence: {result['confidence']:.1%})"
        
        analysis += f"""

Consensus Analysis:
- {spam_votes}/{total_models} models detected spam
- Average confidence: {avg_confidence:.1%}
- Final verdict: {'SPAM/PHISHING' if consensus_spam else 'LEGITIMATE'}

Recommendation: {'DO NOT trust this email. Do not click links or provide personal information.' if consensus_spam else 'This email appears to be legitimate based on the analysis.'}
"""
        
        return {
            'success': True,
            'analysis': analysis,
            'detailed_results': results,
            'consensus': {
                'is_spam': consensus_spam,
                'spam_votes': spam_votes,
                'total_models': total_models,
                'average_confidence': avg_confidence
            }
        }