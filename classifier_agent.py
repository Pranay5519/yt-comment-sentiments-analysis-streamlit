from pydantic import BaseModel, Field
from typing import TypedDict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv  import load_dotenv
load_dotenv()

class TopicState(TypedDict):
    comments: List[str]
    topics: List[str]
    classified_comments: List[dict]
    
class TopicDiscoveryOutput(BaseModel):
    topics: List[str] = Field(
        description="Short, non-overlapping topic names"
    )
    
class ClassifiedComment(BaseModel):
    comment: str
    topic: str


class TopicClassificationOutput(BaseModel):
    results: List[ClassifiedComment]
    
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0
)

topic_discovery_llm = llm.with_structured_output(TopicDiscoveryOutput)
topic_classification_llm = llm.with_structured_output(TopicClassificationOutput)

def discover_topics(state: TopicState):
    comments_text = "\n".join(state["comments"][:100])

    prompt = f"""
You are analyzing YouTube comments.

TASK:
- Create discussion topics from the comments
- Topics must be created by you
- 2–3 words per topic
- No sentiment words
- Max 8 topics
- Dont Include  "Other" as a topic
COMMENTS:
{comments_text}
"""

    response = topic_discovery_llm.invoke(prompt)

    return {
        **state,
        "topics": response.topics
    }
def classify_comments(state: TopicState):
    comments_text = "\n".join(state["comments"])
    topics_text = ", ".join(state["topics"])

    prompt = f"""
You are classifying comments into topics.

TOPICS:
{topics_text}

RULES:
- Use ONLY the provided topics
- One topic per comment
- Do not invent new topics
- No explanations

COMMENTS:
{comments_text}
"""

    response = topic_classification_llm.invoke(prompt)

    return {
        **state,
        "classified_comments": [
            item.model_dump() for item in response.results
        ]
    }
from langgraph.graph import StateGraph , START , END

graph = StateGraph(TopicState)

graph.add_node("discover_topics", discover_topics)
graph.add_node("classify_comments", classify_comments)

graph.add_edge(START , "discover_topics")
graph.add_edge("discover_topics", "classify_comments")
graph.add_edge("classify_comments" , END)
topic_graph = graph.compile()
