from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from crewai_tools import PDFSearchTool

from tools.tool import DocumentSearchTool

pdf_tool = DocumentSearchTool(pdf='/Users/akshaypachaar/Eigen/ai-engineering/agentic_rag/knowledge/dspy.pdf')
web_search_tool = SerperDevTool()


@CrewBase
class AgenticRag():
	"""AgenticRag crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'


	@agent
	def retriever_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['retriever_agent'],
			verbose=True,
			tools=[
				pdf_tool,
				web_search_tool
			]
		)
