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

	agents_config = '/Users/darky/Documents/mat-ml/agentic-rag/src/config/agents.yaml'
	tasks_config = '/Users/darky/Documents/mat-ml/agentic-rag/src/config/task.yaml'

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

	@agent
	def response_synthesizer_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['response_synthesizer_agent'],
			verbose=True
		)


	@task
	def retrieval_task(self) -> Task:
		return Task(
			config=self.tasks_config['retrieval_task'],
		)

	@task
	def response_task(self) -> Task:
		return Task(
			config=self.tasks_config['response_task'],
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the AgenticRag crew"""

		return Crew(
			agents=self.agents, 
			tasks=self.tasks, 
			process=Process.sequential,
			verbose=True,
		)

