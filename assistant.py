import os
import json
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import StructuredTool
import importlib
import inspect
from skills.basic_skill import BasicSkill
from dotenv import load_dotenv
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log, retry_if_exception_type

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
        skill_descriptions = "\n".join([f"- {name}: {skill.metadata['description']}" for name, skill in self.known_skills.items()])
        return f"""
        You are a personal assistant with access to various skills. Your primary function is to assist users by utilizing these skills whenever possible. Always consider using a skill before generating information on your own.

        Available skills:
        {skill_descriptions}

        When a user's request aligns with a skill's functionality, prioritize using that skill. If multiple skills are relevant, use them in combination. Only generate information yourself when no skill is applicable or when additional context is needed.

        Remember:
        1. Always prefer using skills over generating information.
        2. If a user explicitly mentions a skill, use it.
        3. Combine multiple skills when necessary to fulfill complex requests.
        4. Provide clear explanations of which skills you're using and why.

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
            ("system", "Now, considering the user's input and your available skills, determine which skill(s) to use. If no skill is directly applicable, explain why and proceed with generating a response. Always aim to use skills when possible.")
        ])
        
        agent = create_openai_tools_agent(self.chatbot, tools, prompt)
        return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception)),
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
            return response
        except Exception as e:
            logger.error(f"Error occurred while getting response: {str(e)}")
            raise  # This will trigger a retry

    def chat(self, user_input, conversation_history):
        logger.info(f"Received user input: {user_input}")
        conversation_history.append(HumanMessage(content=user_input))
        try:
            response = self.get_response(conversation_history)
            logger.info(f"Generated response: {response}")
            
            if 'tool_calls' in response and len(response['tool_calls']) > 1:
                # Use multi-skill logic
                return self.handle_multi_skill(response, conversation_history)
            elif 'tool_calls' in response and len(response['tool_calls']) == 1:
                # Use single-skill logic
                return self.handle_single_skill(response, conversation_history)
            else:
                # No skill calls, just return the response
                conversation_history.append(AIMessage(content=response['output']))
                return response['output'], conversation_history
        except Exception as e:
            logger.error(f"Failed to generate response after retries: {str(e)}")
            return "I'm sorry, but I'm having trouble connecting to my knowledge base right now. Please try again later or ask me something else.", conversation_history

    def handle_single_skill(self, response, conversation_history):
        tool_call = response['tool_calls'][0]
        skill_name = tool_call['name']
        skill_to_call = self.known_skills.get(skill_name)
        if skill_to_call:
            function_args = json.loads(tool_call['arguments'])
            logger.info(f"Calling {skill_name} with arguments {function_args}")
            skill_response = str(skill_to_call.perform(**function_args))
            logger.info(f"Skill Response: {skill_response}")
            conversation_history.append(ToolMessage(content=skill_response, tool_call_id=tool_call['id']))
            final_response = self.get_response(conversation_history)
            conversation_history.append(AIMessage(content=final_response['output']))
            return final_response['output'], conversation_history
        else:
            error_message = f"Skill {skill_name} not found."
            logger.error(error_message)
            return error_message, conversation_history

    def handle_multi_skill(self, response, conversation_history):
        tool_calls = response['tool_calls']
        logger.info(f"Multiple tool calls detected: {len(tool_calls)}")
        
        for tool_call in tool_calls:
            skill_name = tool_call['name']
            skill_to_call = self.known_skills.get(skill_name)
            if skill_to_call:
                function_args = json.loads(tool_call['arguments'])
                
                # Get the function signature and call the function with given arguments
                sig = inspect.signature(skill_to_call.perform)
                call_args = {
                    k: function_args.get(k, v.default)
                    for k, v in sig.parameters.items()
                    if k in function_args or v.default is not inspect.Parameter.empty
                }
                logger.info(f"Calling {skill_name} with arguments {call_args}")
                
                skill_response = str(skill_to_call.perform(**call_args))
                
                logger.info(f"Skill Response: {skill_response}")

                # Add skill response to conversation history
                conversation_history.append(ToolMessage(content=skill_response, tool_call_id=tool_call['id']))
        
        # Get a new response from the model with all skill outputs
        final_response = self.get_response(conversation_history)
        conversation_history.append(AIMessage(content=final_response['output']))
        return final_response['output'], conversation_history