from skills.basic_skill import BasicSkill
import os
import json
import shutil
import random
import re
from datetime import datetime, timedelta
import autogen
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import logging
import traceback

load_dotenv()


class DynamicSimulationSkill(BasicSkill):
    def __init__(self):
        self.name = 'DynamicSimulation'
        self.metadata = {
            'name': self.name,
            'description': 'Creates a dynamic AutoGen-based simulation with interactive agents, environment variable tracking, time span monitoring, and inter-agent interactions.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'simulation_name': {
                        'type': 'string',
                        'description': 'A unique name for the simulation.'
                    },
                    'goal': {
                        'type': 'string',
                        'description': 'A detailed description of the simulation\'s objective.'
                    },
                    'agents': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'A list of agent names for the simulation.'
                    },
                    'environment_variables': {
                        'type': 'object',
                        'description': 'A dictionary of initial values for all environment variables in the simulation.'
                    },
                    'time_span': {
                        'type': 'object',
                        'properties': {
                            'start': {'type': 'string', 'format': 'date-time'},
                            'end': {'type': 'string', 'format': 'date-time'},
                            'step': {'type': 'string', 'format': 'duration'}
                        },
                        'description': 'Configuration for the time span of the simulation.'
                    },
                    'interaction_matrix': {
                        'type': 'object',
                        'description': 'A dictionary defining which agents can interact with each other.'
                    },
                    'max_turns': {
                        'type': 'integer',
                        'default': 100,
                        'description': 'The maximum number of turns or iterations the simulation will run.'
                    },
                    'num_simulations': {
                        'type': 'integer',
                        'default': 1,
                        'description': 'The number of times to run the simulation.'
                    },
                    'temperature': {
                        'type': 'number',
                        'default': 0.7,
                        'description': 'The temperature setting for the language model.'
                    },
                    'max_tokens': {
                        'type': 'integer',
                        'default': 2000,
                        'description': 'The maximum number of tokens the language model should generate in each response.'
                    }
                },
                'required': ['simulation_name', 'goal', 'agents', 'environment_variables', 'time_span', 'interaction_matrix']
            }
        }
        super().__init__(name=self.name, metadata=self.metadata)

        self.groq_api_key = os.getenv('GROQ_API_KEY')
        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
        self.model = os.getenv(
            'LLM_MODEL', 'llama3-groq-70b-8192-tool-use-preview')
        self.chatbot = ChatGroq(
            model=self.model, groq_api_key=self.groq_api_key)

        self.simulation_log_file = "simulation_log.json"
        self.setup_logging()

    def setup_logging(self):
        log_dir = "simulation_logs"
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(filename=os.path.join(log_dir, 'simulation_process.log'),
                            level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def sanitize_name(self, name):
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', name)
        if not sanitized or not sanitized[0].isalpha() and sanitized[0] != '_':
            sanitized = '_' + sanitized
        return sanitized

    def perform(self, simulation_name, goal, agents, environment_variables, time_span, interaction_matrix, max_turns=100, num_simulations=1, temperature=0.7, max_tokens=2000):
        try:
            logging.info(f"Starting simulation: {simulation_name}")
            logging.info(f"Goal: {goal}")

            simulations_dir = os.path.join(os.getcwd(), "simulations")
            os.makedirs(simulations_dir, exist_ok=True)

            sim_dir = os.path.join(
                simulations_dir, self.sanitize_name(simulation_name))
            os.makedirs(sim_dir, exist_ok=True)

            # Convert time_span to the required format
            if isinstance(time_span, str):
                # Assume the input is in the format "X days"
                num_days = int(time_span.split()[0])
                start_date = datetime.now()
                end_date = start_date + timedelta(days=num_days)
                formatted_time_span = {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                    "step": "1 day"
                }
            elif isinstance(time_span, dict):
                # Ensure the dates are in the correct format
                formatted_time_span = {
                    "start": datetime.fromisoformat(time_span['start'].replace('Z', '+00:00')).isoformat(),
                    "end": datetime.fromisoformat(time_span['end'].replace('Z', '+00:00')).isoformat(),
                    "step": time_span.get('step', "1 day")
                }
            else:
                raise ValueError(
                    "Invalid time_span format. Expected string or dictionary.")

            sim_config = {
                'simulation_name': simulation_name,
                'goal': goal,
                'agents': [{'name': agent, 'role': 'Participant', 'prompt': f'You are {agent}. Your goal is to {goal}', 'influence_areas': list(environment_variables.keys())} for agent in agents],
                'environment_variables': environment_variables,
                'time_span': formatted_time_span,
                'interaction_matrix': {agent: [other_agent for other_agent in agents if other_agent != agent] for agent in agents},
                'max_turns': max_turns,
                'num_simulations': num_simulations,
                'temperature': temperature,
                'max_tokens': max_tokens
            }

            config_file = os.path.join(sim_dir, "sim_config.json")
            with open(config_file, 'w') as file:
                json.dump(sim_config, file, indent=2)

            logging.info("Simulation configuration saved")

            # Create simulation script
            sim_script = self.generate_simulation_script(sim_config)
            script_path = os.path.join(sim_dir, "run_simulation.py")
            with open(script_path, 'w', encoding='utf-8') as file:
                file.write(sim_script)
            logging.info("Simulation script created")

            # Run simulations
            simulation_results = self.run_simulations(sim_config)
            self.log_simulation(
                simulation_name, sim_config, simulation_results)
            logging.info(f"Completed {num_simulations} simulation runs")

            analysis = self.analyze_results(simulation_results)
            logging.info("Simulation results analyzed")

            # Generate and save report
            report = self.generate_report(
                simulation_name, sim_config, simulation_results, analysis)
            report_path = self.save_report(simulation_name, report)
            logging.info(f"Simulation report saved at: {report_path}")

            return f"""
            The {simulation_name} simulation has been created and executed successfully.
            Simulation files are located in: {sim_dir}
            You can re-run the simulation manually by executing the 'run_simulation.py' script in that directory.

            Simulation Analysis Summary:
            {json.dumps(analysis, indent=2)}

            Full report saved at: {report_path}
            """
        except Exception as e:
            logging.error(
                f"An error occurred during simulation: {str(e)}", exc_info=True)
            return f"An error occurred during simulation: {str(e)}\nError details: {traceback.format_exc()}"

    def generate_simulation_script(self, sim_config):
        sim_dir = os.path.join(os.getcwd(), "simulations",
                               self.sanitize_name(sim_config['simulation_name']))
        config_path = os.path.join(sim_dir, "sim_config.json")
        results_path = os.path.join(sim_dir, "simulation_results.json")

        return f"""
import json
import os
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import autogen

load_dotenv()

class SimulationEnvironment:
    def __init__(self, initial_state, time_span):
        self.state = initial_state
        self.current_time = datetime.fromisoformat(time_span['start'])
        self.end_time = datetime.fromisoformat(time_span['end'])
        self.time_step = self.parse_time_step(time_span['step'])

    def parse_time_step(self, step_string):
        number, unit = step_string.split()
        number = int(number)
        if unit in ['second', 'seconds']:
            return timedelta(seconds=number)
        elif unit in ['minute', 'minutes']:
            return timedelta(minutes=number)
        elif unit in ['hour', 'hours']:
            return timedelta(hours=number)
        elif unit in ['day', 'days']:
            return timedelta(days=number)
        else:
            raise ValueError(f"Unsupported time unit: {{unit}}")

    def update(self, changes):
        for key, value in changes.items():
            if key in self.state:
                self.state[key] = value

    def step_time(self):
        self.current_time += self.time_step
        return self.current_time <= self.end_time

    def get_state(self):
        return {{**self.state, 'current_time': self.current_time.isoformat()}}

def create_groq_client():
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    model = os.getenv('LLM_MODEL', 'llama3-groq-70b-8192-tool-use-preview')
    return ChatGroq(model=model, groq_api_key=groq_api_key)

def get_llm_config(temperature, max_tokens):
    return {{
        "temperature": temperature,
        "max_tokens": max_tokens,
    }}

def run_simulation(config):
    client = create_groq_client()
    llm_config = get_llm_config(config['temperature'], config['max_tokens'])

    environment = SimulationEnvironment(config['environment_variables'], config['time_span'])

    agents = []
    for agent_config in config['agents']:
        agent = autogen.AssistantAgent(
            name=agent_config['name'],
            system_message=f"{{agent_config['prompt']}} You can influence these variables: {{', '.join(agent_config['influence_areas'])}}. You can interact with these agents: {{', '.join(config['interaction_matrix'][agent_config['name']])}}",
            llm_config=llm_config,
        )
        agents.append((agent, agent_config['influence_areas']))

    user_proxy = autogen.UserProxyAgent(
        name="SimulationManager",
        system_message="You are managing the simulation. Provide updates on the environment and facilitate interactions between agents.",
        human_input_mode="NEVER",
        llm_config=llm_config,
    )

    group_chat = autogen.GroupChat(agents=[agent for agent, _ in agents] + [user_proxy], messages=[], max_round=config['max_turns'])
    manager = autogen.GroupChatManager(groupchat=group_chat, llm_config=llm_config)

    simulation_history = []

    while environment.step_time() and len(simulation_history) < config['max_turns']:
        current_state = environment.get_state()
        simulation_history.append(current_state)

        state_description = f"Current simulation state: {{json.dumps(current_state, indent=2)}}"
        user_proxy.initiate_chat(manager, message=state_description)

        for agent, influence_areas in agents:
            action_request = f"{{agent.name}}, based on the current state, what actions do you want to take? You can influence: {{', '.join(influence_areas)}}. You can interact with: {{', '.join(config['interaction_matrix'][agent.name])}}"
            response = user_proxy.initiate_chat(agent, message=action_request)
            
            # Process agent's response and update environment
            try:
                action = json.loads(response.last_message()['content'])
                environment.update(action)

                # Handle agent interactions
                for interaction_target in config['interaction_matrix'][agent.name]:
                    if interaction_target in action:
                        target_agent = next(a for a, _ in agents if a.name == interaction_target)
                        interaction_message = f"{{agent.name}} wants to interact with you: {{action[interaction_target]}}"
                        user_proxy.initiate_chat(target_agent, message=interaction_message)

            except json.JSONDecodeError:
                print(f"Warning: Could not parse action from {{agent.name}}")

    return simulation_history

if __name__ == "__main__":
    config_path = "{config_path}"
    results_path = "{results_path}"

    with open(config_path, 'r') as f:
        config = json.load(f)

    results = []
    for _ in range(config['num_simulations']):
        simulation_result = run_simulation(config)
        results.append(simulation_result)

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Simulation completed. Results saved in '{{results_path}}'.")
"""

    def run_simulations(self, sim_config):
        sim_dir = os.path.join(os.getcwd(), "simulations",
                               self.sanitize_name(sim_config['simulation_name']))
        script_path = os.path.join(sim_dir, "run_simulation.py")
        results_path = os.path.join(sim_dir, "simulation_results.json")

        os.system(f"python {script_path}")

        if not os.path.exists(results_path):
            # If results file doesn't exist, create an empty one
            with open(results_path, 'w') as f:
                json.dump([], f)
            logging.warning(
                f"Simulation results file not found. Created an empty file at {results_path}")

        with open(results_path, 'r') as f:
            return json.load(f)

    def analyze_results(self, simulation_results):
        if not simulation_results:
            return {"error": "No simulation results to analyze."}

        analysis = {
            "num_simulations": len(simulation_results),
            "avg_simulation_length": sum(len(sim) for sim in simulation_results) / len(simulation_results) if simulation_results else 0,
            "variable_trends": {},
            "agent_interactions": {}
        }

        # Analyze variable trends
        if simulation_results and simulation_results[0]:
            for var in simulation_results[0][0].keys():
                if var != 'current_time':
                    values = [sim[-1].get(var, 0)
                              for sim in simulation_results]
                    analysis["variable_trends"][var] = {
                        "start_avg": sum(sim[0].get(var, 0) for sim in simulation_results) / len(simulation_results),
                        "end_avg": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values)
                    }

        # Analyze agent interactions
        for sim in simulation_results:
            for state in sim:
                for key, value in state.items():
                    if isinstance(value, dict) and 'interactions' in value:
                        agent = key
                        for interaction in value['interactions']:
                            target = interaction.get('target', 'unknown')
                            if agent not in analysis["agent_interactions"]:
                                analysis["agent_interactions"][agent] = {}
                            if target not in analysis["agent_interactions"][agent]:
                                analysis["agent_interactions"][agent][target] = 0
                            analysis["agent_interactions"][agent][target] += 1

        return analysis

    def generate_report(self, simulation_name, sim_config, simulation_results, analysis):
        return {
            "simulation_name": simulation_name,
            "execution_time": datetime.now().isoformat(),
            "configuration": sim_config,
            "results_summary": {
                "total_simulations": len(simulation_results),
                "average_simulation_length": analysis.get("avg_simulation_length", 0)
            },
            "variable_trends": analysis.get("variable_trends", {}),
            "agent_interactions": analysis.get("agent_interactions", {}),
            "full_simulation_results": simulation_results
        }

    def save_report(self, simulation_name, report):
        reports_dir = "simulation_reports"
        os.makedirs(reports_dir, exist_ok=True)
        report_filename = f"{self.sanitize_name(simulation_name)}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = os.path.join(reports_dir, report_filename)
        with open(report_path, 'w', encoding='utf-8') as file:
            json.dump(report, file, indent=2, ensure_ascii=False)
        return report_path

    def log_simulation(self, simulation_name, sim_config, results):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "simulation_name": simulation_name,
            "config": sim_config,
            "results": results
        }
        try:
            if os.path.exists(self.simulation_log_file):
                with open(self.simulation_log_file, 'r+') as file:
                    log = json.load(file)
                    log.append(log_entry)
                    file.seek(0)
                    json.dump(log, file, indent=2)
            else:
                with open(self.simulation_log_file, 'w') as file:
                    json.dump([log_entry], file, indent=2)
        except Exception as e:
            logging.error(f"Error logging simulation: {str(e)}")

    def backup_simulation(self, simulation_name, backup_dir="simulation_backups"):
        source_dir = os.path.join(
            "simulations", self.sanitize_name(simulation_name))
        if not os.path.exists(source_dir):
            return f"Simulation '{simulation_name}' not found."

        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(
            backup_dir, f"{self.sanitize_name(simulation_name)}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        try:
            shutil.copytree(source_dir, backup_path)
            return f"Simulation '{simulation_name}' has been backed up successfully to {backup_path}."
        except Exception as e:
            return f"An error occurred while backing up the simulation: {str(e)}"

    def restore_simulation(self, backup_path, new_simulation_name=None):
        if not os.path.exists(backup_path):
            return f"Backup at '{backup_path}' not found."

        if new_simulation_name:
            target_dir = os.path.join(
                "simulations", self.sanitize_name(new_simulation_name))
        else:
            target_dir = os.path.join("simulations", os.path.basename(
                backup_path).split('_backup_')[0])

        try:
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            shutil.copytree(backup_path, target_dir)

            # Update simulation name in config if a new name was provided
            if new_simulation_name:
                config_path = os.path.join(target_dir, "sim_config.json")
                with open(config_path, 'r') as f:
                    config = json.load(f)
                config['simulation_name'] = new_simulation_name
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)

            return f"Simulation has been restored successfully to {target_dir}."
        except Exception as e:
            return f"An error occurred while restoring the simulation: {str(e)}"

    def list_simulations(self):
        simulations_dir = "simulations"
        if not os.path.exists(simulations_dir):
            return "No simulations have been created yet."

        simulations = os.listdir(simulations_dir)
        if not simulations:
            return "No simulations have been created yet."

        return "Created Simulations:\n" + "\n".join(simulations)

    def get_simulation_info(self, simulation_name):
        sim_dir = os.path.join(
            "simulations", self.sanitize_name(simulation_name))
        if not os.path.exists(sim_dir):
            return f"Simulation '{simulation_name}' not found."

        config_file = os.path.join(sim_dir, "sim_config.json")
        try:
            with open(config_file, 'r') as file:
                config = json.load(file)

            info = f"Simulation Name: {config['simulation_name']}\n"
            info += f"Goal: {config['goal']}\n"
            info += f"Number of Agents: {len(config['agents'])}\n"
            info += "Agents:\n"
            for agent in config['agents']:
                info += f"  - {agent['name']} ({agent['role']})\n"
                info += f"    Influence Areas: {', '.join(agent['influence_areas'])}\n"
            info += f"Time Span: {config['time_span']['start']} to {config['time_span']['end']} (Step: {config['time_span']['step']})\n"
            info += f"Max Turns: {config['max_turns']}\n"
            info += f"Number of Simulations: {config['num_simulations']}\n"
            info += f"Temperature: {config['temperature']}\n"
            info += f"Max Tokens: {config['max_tokens']}\n"
            return info
        except Exception as e:
            return f"An error occurred while retrieving simulation information: {str(e)}"

    def delete_simulation(self, simulation_name):
        sim_dir = os.path.join(
            "simulations", self.sanitize_name(simulation_name))
        if not os.path.exists(sim_dir):
            return f"Simulation '{simulation_name}' not found."

        try:
            shutil.rmtree(sim_dir)
            return f"Simulation '{simulation_name}' has been deleted successfully."
        except Exception as e:
            return f"An error occurred while deleting the simulation: {str(e)}"

# End of DynamicSimulationSkill class
