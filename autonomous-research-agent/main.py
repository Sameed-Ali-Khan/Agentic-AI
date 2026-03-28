import os
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

os.environ["GROQ_API_KEY"] = "api key removed to upload on git"

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.6)

search_tool = DuckDuckGoSearchRun()
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=3))

tools = [search_tool, wiki_tool]

template = """You are a helpful research assistant. Use the following tools to gather information.

{tools}

Use this exact format:

Question: the input question you must answer
Thought: I need to...
Action: the action to take (must be one of [{tool_names}])
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Final Answer: [your final response here]

VERY IMPORTANT: 
- After gathering enough information, you MUST output "Final Answer:" followed by the full report.
- Do NOT output anything after the Final Answer section.
- The Final Answer must be in this exact markdown format:

**Cover Page**

**Title:** Impact of AI in Healthcare
**Autonomous Research Agent**
**Generated on:** {current_date}

**Title**
Impact of AI in Healthcare

**Introduction**
[Write 2-3 good paragraphs]

**Key Findings**
- Point 1
- Point 2
- Point 3
- Point 4
- Point 5

**Challenges**
- Challenge 1
- Challenge 2
- Challenge 3

**Future Scope**
- Opportunity 1
- Opportunity 2
- Opportunity 3

**Conclusion**
[Write 2 strong paragraphs]

Begin!

Question: {input}
{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=12, handle_parsing_errors=True)

def generate_research_report(topic):
    current_date = datetime.now().strftime("%B %d, %Y")
    query = f"Provide a detailed research report on: {topic}. Include latest statistics, key findings, challenges, and future scope."
    
    result = agent_executor.invoke({
        "input": query, 
        "current_date": current_date
    })
    
    report = result["output"]
    filename = f"report_{topic.replace(' ', '_')[:40]}.md"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n✅ Report successfully saved as: {filename}")
    print(report)

if __name__ == "__main__":
    topic = input("Enter the research topic: ").strip()
    print("\nGenerating report, please wait...\n")
    generate_research_report(topic)