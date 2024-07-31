import os
import json
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import StructuredTool
import importlib
import inspect
from skills.basic_skill import BasicSkill
from dotenv import load_dotenv
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class Assistant:
    def __init__(self):
        self.model = os.getenv('LLM_MODEL', 'llama3-groq-70b-8192-tool-use-preview')
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY not found in environment variables or .env file. "
                "Please set it in your .env file or as an environment variable."
            )
        self.chatbot = ChatGroq(model=self.model, groq_api_key=self.groq_api_key)
        self.known_skills = self.load_skills()
        self.system_message = self.create_system_message()
        self.agent_executor = self.create_agent_executor()

    def load_skills(self):
        skills = {}
        skills_dir = 'skills'
        for filename in os.listdir(skills_dir):
            if filename.endswith('.py') and filename != 'basic_skill.py':
                module_name = filename[:-3]
                module = importlib.import_module(f'skills.{module_name}')
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and issubclass(obj, BasicSkill) and obj != BasicSkill:
                        skill = obj()
                        skills[skill.name] = skill
        logger.info(f"Loaded skills: {', '.join(skills.keys())}")
        return skills

    def create_system_message(self):
        return f"""
        You are a personal assistant who helps manage tasks using various skills.
        You never give IDs to the user since those are just for you to keep track of.
        When a user asks to perform an action and you don't have enough information, clarify with the user.
        The current date is: {datetime.now().date()}
        """

    def create_agent_executor(self):
        tools = [
            skill.tool if hasattr(skill, 'tool') else StructuredTool.from_function(
                func=skill.perform,
                name=skill.name,
                description=skill.metadata.get('description', '')
            )
            for skill in self.known_skills.values()
        ]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_tools_agent(self.chatbot, tools, prompt)
        return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        before_sleep=before_sleep_log(logger, logging.INFO)
    )
    def get_response(self, messages):
        logger.info("Attempting to get response from Groq API")
        chat_history = messages[1:-1]  # Exclude system message and latest human message
        human_message = messages[-1].content
        try:
            response = self.agent_executor.invoke({
                "input": human_message,
                "chat_history": chat_history
            })
            logger.info("Successfully received response from Groq API")
            return response["output"]
        except Exception as e:
            logger.error(f"Error occurred while getting response: {str(e)}")
            logger.info("Retrying...")
            raise  # This will trigger a retry

    def chat(self, user_input, conversation_history):
        logger.info(f"Received user input: {user_input}")
        conversation_history.append(HumanMessage(content=user_input))
        try:
            response = self.get_response(conversation_history)
            logger.info(f"Generated response: {response}")
            conversation_history.append(AIMessage(content=response))
            return response, conversation_history
        except Exception as e:
            logger.error(f"Failed to generate response after retries: {str(e)}")
            return "I'm sorry, but I'm having trouble connecting to my knowledge base right now. Please try again later.", conversation_history