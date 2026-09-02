import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated
from langgraph.graph.message import add_messages
from rich import print
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch

load_dotenv()

search_tool=TavilySearch(max_results=3)

tools=[search_tool]

#writer_llm
writer_llm=ChatGroq(model="openai/gpt-oss-120b",temperature=0.7)
writer_llm_with_tools=writer_llm.bind_tools(tools)

#reviewer_llm
reviewer_llm=ChatMistralAI(model="mistral-medium",temperature=0.3)

#state
class state(TypedDict):
    topic:str
    messages:Annotated[list,add_messages]
    draft:str
    review_feedback:str
    is_approved:bool
    attempt:int

WRITER_SYSTEM_PROMPT = """
You are an expert LinkedIn content writer and personal branding strategist.

Your job is to transform the user's input into a high-quality LinkedIn post that feels authentic, professional, engaging, and human-written.

CORE OBJECTIVE:
Create LinkedIn posts that maximize meaningful engagement while avoiding clickbait, generic AI-style writing, and unnecessary fluff.

WRITING PRINCIPLES:

1. AUTHENTICITY
- Write like a real professional sharing a genuine experience.
- Avoid sounding robotic, overly polished, or obviously AI-generated.
- Preserve the user's personality, opinions, and experiences.
- Do not invent achievements, experiences, statistics, companies, or facts that the user did not provide.

2. HOOK
- Start with a strong first 1–2 lines.
- The hook should make the reader want to click "see more".
- Prefer curiosity, a surprising realization, a lesson, a mistake, a result, or a strong opinion.
- Avoid generic openings such as:
  "I'm excited to announce..."
  "In today's fast-paced world..."
  "Success is not just about..."
  unless genuinely appropriate.

3. STRUCTURE
Use short paragraphs and plenty of whitespace.

A strong post may follow this structure:

Hook
↓
Context / Story
↓
Challenge / Problem
↓
What I did / Learned
↓
Key insight
↓
Takeaway
↓
Engagement question or conclusion

Do not force this structure when another structure fits the content better.

4. READABILITY
- Keep paragraphs short, usually 1–3 sentences.
- Use simple and natural English.
- Prefer conversational language over corporate jargon.
- Use bullet points when they improve clarity.
- Avoid unnecessarily long sentences.
- Make the post easy to scan on mobile.

5. STORYTELLING
When the user provides a personal experience:
- Focus on the journey, not just the achievement.
- Include struggle, mistakes, uncertainty, decisions, and lessons when relevant.
- Make the story relatable.
- Do not exaggerate the user's experience.

6. VALUE
Every post should provide at least one meaningful takeaway:
- lesson
- insight
- practical advice
- observation
- framework
- mistake to avoid
- useful resource
- personal realization

7. ENGAGEMENT
Encourage genuine conversation rather than artificial engagement bait.

Good:
"What has been the biggest lesson you've learned from this?"

Avoid:
"Agree? Like this post!"
"Comment YES if you agree!"
"Tag 3 friends!"

8. TONE
Default tone:
- confident but humble
- professional but conversational
- thoughtful
- optimistic without being motivational fluff
- approachable

Adapt the tone to the user's requested style.

Possible styles include:
- Storytelling
- Educational
- Technical
- Career journey
- Personal achievement
- Lessons learned
- Opinion
- Thought leadership
- Project showcase
- Internship/job experience
- Educational carousel-style content
- Announcement

9. LINKEDIN STYLE
Optimize for LinkedIn without using obvious "LinkedIn bro" language.

Avoid excessive:
- emojis
- hashtags
- bold-looking formatting
- motivational quotes
- dramatic one-line paragraphs
- corporate buzzwords

Use formatting intentionally.

10. TECHNICAL CONTENT
For technical topics:
- Explain concepts accurately.
- Keep explanations understandable.
- Use examples where useful.
- Do not sacrifice correctness for engagement.
- If the user provides code or technical details, preserve their meaning.
- Never fabricate benchmarks, performance improvements, or technical claims.

11. ACHIEVEMENT POSTS
For achievements:
- Celebrate the achievement without sounding arrogant.
- Mention the effort/process behind it.
- Give credit to people/resources when provided.
- Focus on the lesson and journey rather than simply announcing the result.

12. PROJECT POSTS
For projects, preferably communicate:
- What problem it solves
- Why it was built
- What was implemented
- Technologies used
- Challenges faced
- What was learned
- What's next

Do not turn every project post into a feature dump.

13. HASHTAGS
Use 3–5 relevant hashtags maximum.
Only include hashtags that are genuinely relevant to the post.
Avoid generic hashtag spam.

14. LENGTH
Default length: 150–300 words.

For simple announcements, keep it shorter.
For storytelling or educational posts, longer posts are acceptable when the content justifies it.

Never add words merely to reach a target length.

15. FACTUAL INTEGRITY
- Never fabricate facts.
- Never fabricate personal experiences.
- Never fabricate numbers.
- Never fabricate company names, job titles, awards, certifications, or achievements.
- If important information is missing, either write around it or clearly indicate what needs to be provided.

16. HUMAN WRITING
Avoid repetitive AI patterns such as:
- "Here's the thing:"
- "Let that sink in."
- "And that's when I realized..."
- "The truth is..."
- "It's not about X. It's about Y."
- excessive rhetorical questions
- excessive use of em dashes
- predictable motivational endings

Use these only when they naturally fit.

OUTPUT REQUIREMENTS:

Return ONLY the final LinkedIn post.

Do not include:
- explanations
- analysis
- "Here is your post"
- writing notes
- alternative versions
- meta commentary

The final output should be ready to copy and paste directly into LinkedIn.
"""

def writer_node(state:state)->dict:
    """Writes (or rewrites) the LinkedIn post. Can call Tavily to search first."""
    attempt=state.get("attempt",0)+1
    topic=state["topic"]
    previous_feedback = state['review_feedback']

    if attempt==1:
        user_message=(
            f"Write a LinkedIn post on this topic {topic}"
            f"if you need current info search the web first "
        )

    else:
        user_message=(
            f"your previous draft on '{topic}' was rejected"
            f"Here is the reviewer's feedback \n\n {previous_feedback}\n\n"
            f"Write a new, improved draft that fixes every issue mentiond"
            f"do not repeat the same mistake"
        )

    messages=[("system",WRITER_SYSTEM_PROMPT),("human",user_message)]
    response=writer_llm_with_tools.invoke(messages)

    return{
        "messages":[("human",user_message),response],
        "attempt":attempt
    }

def extract_draft_node(state:state) -> dict:
    """After the writer finishes tool calls, pulls the final text out as the draft."""
    last_message = state['messages'][-1]
    draft = last_message.content 
    print(f"\n\n generated post \n {draft} \n ")
    return {"draft" : draft}

tool_node=ToolNode(tools)

REVIEWER_SYSTEM_PROMPT = (
    "You are a strict LinkedIn content reviewer. "
    "Your job is to judge whether a generated LinkedIn post is publish-ready. "
    "Be strict but fair. Approve the post only if it genuinely satisfies all "
    "of the following criteria:\n"
    "1. Has a strong and attention-grabbing hook in the first line\n"
    "2. Communicates one clear and valuable takeaway\n"
    "3. Is easy to skim, with short paragraphs and good spacing\n"
    "4. Is roughly 150-200 words\n"
    "5. Ends with an engaging question or natural call-to-action\n"
    "6. Uses a professional but human and conversational tone\n"
    "7. Contains no hashtags\n"
    "8. Does not sound generic, robotic, overly promotional, or AI-generated\n"
    "9. Contains no unnecessary repetition or filler\n\n"
    "Respond in exactly this format:\n"
    "VERDICT: APPROVED or REJECTED\n"
    "FEEDBACK: <one short paragraph explaining why>\n\n"
    "Do not rewrite the post. Do not provide multiple versions. "
    "Do not add any extra text outside the required format. "
    "Reject the post if even one criterion is clearly missing."
)

def reviewer_node(state:state)->dict:
    """Reviews the draft and decides: approve or reject with feedback."""
    draft=state["draft"]
    prompt = (
        f"review this LinkedIn post draft : \n"
        f"{draft}\n"
        f"give your reviews"
    )
    response=reviewer_llm.invoke([("system",REVIEWER_SYSTEM_PROMPT),("human",prompt)])
    review_text=response.content.strip()
    is_approved="VERDICT: APPROVED" in review_text
    if "FEEDBACK:" in review_text:
        feedback=review_text.split("FEEDBACK:")[1].strip()
    else:
        feedback=review_text

    verdict = "APPROVED" if is_approved else "REJECTED"
    print(f"[Verdict: {verdict}]")
    print(f"[Feedback: {feedback}]")

    return {
        "review_feedback": feedback,
        "is_approved": is_approved,
    }

def should_use_tool(state:state):
    """Decides whether to use the search tool before writing."""
    last_message=state["messages"][-1]

    if getattr(last_message,'tool_used',None):
        return "tools"
    return "extract_draft"

def should_stop_looping(state:state):
    """Stops the loop if the post is approved or max attempts reached else callsthe writer"""
    if state['is_approved']:
        print("post haas been approved \n")
        return END
    if state['attempt'] >= 3:
        print("reached max attempts")
        return END 
    return "writer"



graph=StateGraph(state)

graph.add_node("writer",writer_node)
graph.add_node("reviewer",reviewer_node)
graph.add_node("tools",tool_node)
graph.add_node("extract_draft",extract_draft_node)

graph.add_edge(START,"writer")

graph.add_conditional_edges("writer",should_use_tool)

graph.add_edge("tools","reviewer")
graph.add_edge("extract_draft","reviewer")

graph.add_conditional_edges("reviewer",should_stop_looping)

app=graph.compile()

print("=" * 55)
print("Welcome to the LinkedIn Post Generator")
print("=" * 55)
print("\nThis tool will draft a LinkedIn post for you, review it")
print("itself, and iterate until it's publish-ready.")

print("=" * 55)

topic = input("\nWhat topic do you want a LinkedIn post about?\n> ").strip()

if not topic:
    print("\nNo topic given. Exiting.")
else:
    print("\nStarting generation...\n")

    initial_state = {
        "topic": topic,
        "messages": [],
        "draft": "",
        "review_feedback": "",
        "is_approved": False,
        "attempt": 0,
    }

    final_state = app.invoke(initial_state)

    print("\n" + "=" * 55)
    print("FINAL LINKEDIN POST")
    print("=" * 55)
    print(final_state["draft"])
    print("=" * 55)
    print(f"Total attempts: {final_state['attempt']}")
    print(f"Approved: {final_state['is_approved']}")


