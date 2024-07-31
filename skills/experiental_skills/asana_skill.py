from skills.basic_skill import BasicSkill
import asana
from asana.rest import ApiException
import json
import os
from datetime import datetime, timedelta

class AsanaSkill(BasicSkill):
    def __init__(self):
        self.name = 'AsanaSkill'
        self.metadata = {
            "name": self.name,
            "description": "Manages tasks and projects in Asana",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create_project", "create_task", "get_projects", "get_tasks"],
                        "description": "The Asana action to perform"
                    },
                    "project_name": {
                        "type": "string",
                        "description": "The name of the project (for create_project action)"
                    },
                    "task_name": {
                        "type": "string",
                        "description": "The name of the task (for create_task action)"
                    },
                    "project_gid": {
                        "type": "string",
                        "description": "The GID of the project (for create_task and get_tasks actions)"
                    },
                    "due_on": {
                        "type": "string",
                        "description": "The due date for the task or project (format: YYYY-MM-DD)"
                    }
                },
                "required": ["action"]
            }
        }
        super().__init__(name=self.name, metadata=self.metadata)
        
        configuration = asana.Configuration()
        configuration.access_token = os.getenv('ASANA_ACCESS_TOKEN', '')
        self.api_client = asana.ApiClient(configuration)
        self.projects_api = asana.ProjectsApi(self.api_client)
        self.tasks_api = asana.TasksApi(self.api_client)
        self.workspace_gid = os.getenv("ASANA_WORKPLACE_ID", "")

    def perform(self, action, **kwargs):
        """
        Perform the specified Asana action with the given parameters.

        Args:
            action (str): The Asana action to perform.
            **kwargs: Additional parameters for the action.

        Returns:
            str: The result of the Asana action.
        """
        actions = {
            "create_project": self.create_project,
            "create_task": self.create_task,
            "get_projects": self.get_projects,
            "get_tasks": self.get_tasks
        }
        if action not in actions:
            return f"Invalid action: {action}. Valid actions are: {', '.join(actions.keys())}"
        return actions[action](**kwargs)

    # The rest of the methods (create_project, create_task, get_projects, get_tasks) remain the same
    # ...