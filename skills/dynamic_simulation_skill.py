from skills.basic_skill import BasicSkill
import os
import json
import shutil
import random
import re
from datetime import datetime
import autogen
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import logging

load_dotenv()

class EnhancedDynamicSimulationSkill(BasicSkill):
    def __init__(self):
        self.name = 'EnhancedDynamicSimulation'
        self.metadata = {
            'name': self.name,
            'description': 'Creates a dynamic AutoGen-based group chat simulation with atomized agents, incorporating probabilistic outcomes based on past experiences.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'simulation_name': {
                        'type': 'string',
                        'description': 'The name of the simulation'
                    },
                    'goal': {
                        'type': 'string',
                        'description': 'The overall goal of the simulation'
                    },
                    'required_agents': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'name': {'type': 'string'},
                                'role': {'type': 'string'},
                                'prompt': {'type': 'string'}
                            },
                            'required': ['name', 'role', 'prompt']
                        },
                        'description': 'List of required agent roles for the simulation'
                    },
                    'inputs': {
                        'type': 'object',
                        'description': 'Input parameters for the simulation'
                    },
                    'expected_outputs': {
                        'type': 'array',
                        'items': {
                            'type': 'string'
                        },
                        'description': 'List of expected output types from the simulation'
                    },
                    'max_turns': {
                        'type': 'integer',
                        'description': 'The maximum number of conversation turns',
                        'default': 10
                    },
                    'num_simulations': {
                        'type': 'integer',
                        'description': 'Number of simulation runs to perform',
                        'default': 1
                    },
                    'temperature': {
                        'type': 'number',
                        'description': 'The temperature setting for the LLM',
                        'default': 0.7
                    },
                    'max_tokens': {
                        'type': 'integer',
                        'description': 'The maximum number of tokens for each LLM response',
                        'default': 2000
                    }
                },
                'required': ['simulation_name', 'goal', 'required_agents', 'inputs', 'expected_outputs']
            }
        }
        super().__init__(name=self.name, metadata=self.metadata)
        
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
        self.model = os.getenv('LLM_MODEL', 'llama3-groq-70b-8192-tool-use-preview')
        self.chatbot = ChatGroq(model=self.model, groq_api_key=self.groq_api_key)
        
        self.memory_file = "memory.json"
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

    def perform(self, simulation_name, goal, required_agents, inputs, expected_outputs, max_turns=10, num_simulations=1, temperature=0.7, max_tokens=2000):
        try:
            logging.info(f"Starting simulation: {simulation_name}")
            logging.info(f"Goal: {goal}")

            simulations_dir = "simulations"
            os.makedirs(simulations_dir, exist_ok=True)

            sim_dir = os.path.join(simulations_dir, simulation_name.lower().replace(' ', '_'))
            os.makedirs(sim_dir, exist_ok=True)

            sim_config = {
                'simulation_name': simulation_name,
                'goal': goal,
                'required_agents': required_agents,
                'inputs': inputs,
                'expected_outputs': expected_outputs,
                'max_turns': max_turns,
                'num_simulations': num_simulations,
                'temperature': temperature,
                'max_tokens': max_tokens
            }

            config_file = os.path.join(sim_dir, "sim_config.json")
            with open(config_file, 'w') as file:
                json.dump(sim_config, file, indent=2)

            logging.info("Simulation configuration saved")

            # Generate agent configurations
            agents = self.generate_agent_configs(required_agents, goal, inputs, expected_outputs)
            logging.info(f"Generated configurations for {len(agents)} agents")

            # Create simulation script
            sim_script = self.generate_simulation_script(agents, sim_config)
            script_path = os.path.join(sim_dir, "run_simulation.py")
            with open(script_path, 'w', encoding='utf-8') as file:
                file.write(sim_script)
            logging.info("Simulation script created")

            # Run simulations
            past_experiences = self.load_memories()
            simulation_results = self.run_simulations(sim_config, past_experiences)
            self.log_simulation(simulation_name, sim_config, simulation_results)
            logging.info(f"Completed {num_simulations} simulation runs")

            analysis = self.analyze_results(simulation_results)
            logging.info("Simulation results analyzed")

            # Generate and save report
            report = self.generate_report(simulation_name, sim_config, simulation_results, analysis)
            report_path = self.save_report(simulation_name, report)
            logging.info(f"Simulation report saved at: {report_path}")

            return f"""
    The {simulation_name} simulation has been created successfully in the '{sim_dir}' directory.
    You can run it manually by executing the 'run_simulation.py' script.

    Simulation Analysis:
    {json.dumps(analysis, indent=2)}

    Full report saved at: {report_path}
    """
        except Exception as e:
            logging.error(f"An error occurred during simulation: {str(e)}", exc_info=True)
            return f"An error occurred during simulation: {str(e)}"

    # ... [keep other existing methods] ...

    def generate_report(self, simulation_name, sim_config, simulation_results, analysis):
        return {
            "simulation_name": simulation_name,
            "execution_time": datetime.now().isoformat(),
            "configuration": sim_config,
            "results_summary": {
                "total_simulations": len(simulation_results),
                "unique_outcomes": len(analysis)
            },
            "analysis": analysis,
            "full_simulation_results": simulation_results
        }

    def save_report(self, simulation_name, report):
        reports_dir = "simulation_reports"
        os.makedirs(reports_dir, exist_ok=True)
        report_filename = f"{simulation_name.lower().replace(' ', '_')}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = os.path.join(reports_dir, report_filename)
        with open(report_path, 'w', encoding='utf-8') as file:
            json.dump(report, file, indent=2, ensure_ascii=False)
        return report_path

    def backup_simulation(self, simulation_name, backup_dir="simulation_backups"):
        source_dir = os.path.join("simulations", simulation_name.lower().replace(' ', '_'))
        if not os.path.exists(source_dir):
            return f"Simulation '{simulation_name}' not found."

        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, f"{simulation_name.lower().replace(' ', '_')}_backup")

        try:
            shutil.copytree(source_dir, backup_path)
            return f"Simulation '{simulation_name}' has been backed up successfully to {backup_path}."
        except Exception as e:
            return f"An error occurred while backing up the simulation: {str(e)}"

    def restore_simulation(self, simulation_name, backup_dir="simulation_backups"):
        backup_path = os.path.join(backup_dir, f"{simulation_name.lower().replace(' ', '_')}_backup")
        if not os.path.exists(backup_path):
            return f"Backup for simulation '{simulation_name}' not found."

        target_dir = os.path.join("simulations", simulation_name.lower().replace(' ', '_'))

        try:
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            shutil.copytree(backup_path, target_dir)
            return f"Simulation '{simulation_name}' has been restored successfully from backup."
        except Exception as e:
            return f"An error occurred while restoring the simulation: {str(e)}"

    # ... [keep other existing methods] ...