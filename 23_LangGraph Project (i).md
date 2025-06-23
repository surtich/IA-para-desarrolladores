# Sidekick

Vamos a crear un proyecto completo con LangGraph que use Multi-Agents.


```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from IPython.display import Image, display
import gradio as gr
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests
import os
from langchain.agents import Tool

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
```

    /home/surtich/projects/IA para desarrolladores/.venv/lib/python3.13/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm



```python
load_dotenv(override=True)
```




    True



### Asynchronous LangGraph

LangGraph puede funcionar en modo síncrono o asíncrono. Lo usaremos más adelante.

To run a tool:  
Sync: `tool.run(inputs)`  
Async: `await tool.arun(inputs)`

To invoke the graph:  
Sync: `graph.invoke(state)`  
Async: `await graph.ainvoke(state)`


```python
class State(TypedDict):
    
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)
```


```python
from typing import Dict
import requests 

def send_email(subject: str, html_body: str, to: str, name: str = None) -> Dict[str, str]:
    """Enviar un correo electrónico"""
    from_email = os.getenv('MAILGUN_FROM')
    to_email = f"{name} <{to}>" if name else to
    content = html_body

    requests.post(
  		f"https://api.mailgun.net/v3/{os.getenv('MAILGUN_SANDBOX')}/messages",
  		auth=("api", os.getenv('MAILGUN_API_KEY')),
  		data={"from": from_email,
			"to": to_email,
  			"subject": subject,
  			"html": content})

    return {"status": "éxito"}

import os
from langchain_community.tools import StructuredTool

tool_send_email = StructuredTool.from_function(send_email, description="Útil para enviar correos electrónicos", name="send_email")



```

El proyecto requiere Playwright instalado. Playwright es una herramienta que permite interactuar con navegadores web de manera programática, lo cual es útil para pruebas automatizadas y scraping.

On Windows and MacOS:  
`playwright install`

On Linux:  
`playwright install --with-Deps chromium`

El código asíncrono en Python funciona como JavaScript, con el uso del `event loop`. Sólo puede haber un `event loop` en una aplicación. En este proyecto vamos a ejecutar código asíncrono dentro del `event loop`, por lo que se requieren `event loops` anidados. Para salvar la limitación de Python hacemos lo siguiente.


```python
import nest_asyncio
nest_asyncio.apply()
```

Hay librería de LangChain que permite usar PlayWright en forma de `tools`


```python

```


```python
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser

async_browser =  create_async_playwright_browser(headless=False)  # headful mode
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
tools = toolkit.get_tools()
```

Empaquetamos las `tools` en un diccionario y probamos una de ellas:


```python
tool_dict = {tool.name:tool for tool in tools}

navigate_tool = tool_dict.get("navigate_browser")
extract_text_tool = tool_dict.get("extract_text")

await navigate_tool.arun({"url": "https://www.cnn.com"})
text = await extract_text_tool.arun({})
```


```python
import textwrap
print(textwrap.fill(text))
```

    Breaking News, Latest News and Videos | CNN CNN values your feedback
    1. How relevant is this ad to you? 2. Did you encounter any technical
    issues? Video player was slow to load content Video content never
    loaded Ad froze or did not finish loading Video content did not start
    after ad Audio on ad was too loud Other issues Ad never loaded Ad
    prevented/slowed the page from loading Content moved around while ad
    loaded Ad was repetitive to ads I've seen previously Other issues
    Cancel Submit Thank You! Your effort and contribution in providing
    this feedback is much
    appreciated. Close Ad Feedback Close icon US World Politics Business
    Health Entertainment Style Travel Sports Science Climate Weather
    Ukraine-Russia War Israel-Hamas War Games More US World Politics
    Business Health Entertainment Style Travel Sports Science Climate
    Weather Ukraine-Russia War Israel-Hamas War Games Watch Listen Live TV
    Subscribe Sign in My Account Settings Newsletters Topics you follow
    Sign out Your CNN account Sign in to your CNN account Sign in My
    Account Settings Newsletters Topics you follow Sign out Your CNN
    account Sign in to your CNN account Live TV Listen Watch Edition US
    International Arabic Español Edition US International Arabic Español
    World Africa Americas Asia Australia China Europe India Middle East
    United Kingdom US Politics Trump Facts First CNN Polls 2025 Elections
    Business Tech Media Calculators Videos Markets Pre-markets After-Hours
    Fear & Greed Investing Markets Now Nightcap Health Life, But Better
    Fitness Food Sleep Mindfulness Relationships Entertainment Movies
    Television Celebrity Tech Innovate Foreseeable Future Mission: Ahead
    Work Transformed Innovative Cities Style Arts Design Fashion
    Architecture Luxury Beauty Video Travel Destinations Food & Drink Stay
    News Videos Sports Football Tennis Golf Motorsport US Sports Olympics
    Climbing Esports Hockey Science Space Life Unearthed Climate Solutions
    Weather Weather Video Climate Ukraine-Russia War Israel-Hamas War
    Features As Equals Call to Earth Freedom Project Impact Your World
    Inside Africa CNN Heroes Watch Live TV CNN Fast Shows A-Z CNN10 CNN
    Max CNN TV Schedules Listen CNN 5 Things Chasing Life with Dr. Sanjay
    Gupta The Assignment with Audie Cornish One Thing Tug of War CNN
    Political Briefing The Axe Files All There Is with Anderson Cooper All
    CNN Audio podcasts Games Daily Crossword Jumble Crossword Sudoblock
    Sudoku 5 Things Quiz About CNN Photos Investigations CNN Profiles CNN
    Leadership CNN Newsletters Work for CNN Follow CNN NATO summit France
    syringe attack Mozambique child abductions Malala Yousafzai Hyper-
    realistic pencil portraits 1,000-year-old sword Ruined castle for $7.5
    million Live Updates 32 min ago THE HAGUE, NETHERLANDS - JUNE 25: U.S.
    President Donald Trump speaks to media at the start of the second day
    of the 2025 NATO Summit on June 25, 2025 in The Hague, Netherlands.
    Among other matters, members are to approve a new defense investment
    plan that raises the target for defense spending to 5% of GDP. (Photo
    by Andrew Harnik/Getty Images) Andrew Harnik/Getty Images Trump in
    Netherlands for NATO summit as Israel-Iran ceasefire appears to hold
    32 min ago Trump maintains Iran strikes caused "total obliteration"
    despite early intel report. Catch up here 1 hr 33 min ago Trump draws
    comparison between US bombing of Hiroshima and his Iran strikes 1 hr
    55 min ago Israeli defense minister designates Iran’s central bank a
    terrorist organization 2 hr 2 min ago Iran may rethink membership of
    key nuclear non-proliferation treaty, foreign minister suggests 2 hr 3
    min ago Trump says he doesn't believe Iran moved enriched uranium
    ahead of US strikes 2 hr 13 min ago Trump maintains Iran strikes
    caused "total obliteration," but acknowledges damage report was
    inconclusive 2 hr 19 min ago Trump says US strikes on Iran nuclear
    sites could help achieve Gaza deal 2 hr 45 min ago Meanwhile, CNN's
    Clarissa Ward reports on the latest on Israel's offensive in Gaza 2 hr
    57 min ago Iran’s parliament votes to suspend cooperation with UN
    nuclear watchdog 1 hr 43 min ago Rutte "totally fine" with Trump
    sharing his fawning text message, stresses US is committed to Article
    5 4 hr 44 min ago NATO chief stands by praise for US attacks on Iran 4
    hr 42 min ago Israel's Smotrich rejects early US military assessment
    on extent of damage to Iran’s nuclear facilities 5 hr 3 min ago White
    House envoy slams Iran intel leak 7 hr ago It’s morning in the Middle
    East. A fragile truce appears to be holding 7 hr 2 min ago Iran
    arrested 700 "mercenaries" working for Israel throughout 12-day
    conflict, state media says 7 hr 8 min ago Seven Israeli soldiers
    killed by blast in Gaza’s Khan Younis, military says 7 hr 28 min ago
    Trump pushes back on CNN report about intel assessment suggesting
    strikes on Iran didn't destroy nuclear sites 7 hr 28 min ago Israel
    will "respond forcefully" to any ceasefire violations, UN ambassador
    says 7 hr 31 min ago Iran's UN ambassador thanks Qatar for its role in
    the ceasefire 7 hr 32 min ago Israel extends mobilization order for
    reservists until July 10 See all updates ( 20 +) • Video 1:47 CNN gets
    an up close look at Israeli airstrike aftermath damage in Iran CNN
    Video CNN gets an up close look at Israeli airstrike aftermath damage
    in Iran 1:47 • Video 1:45 Pool Video ‘A tremendous victory for
    everybody:’ Trump on Iran strikes 1:45 Trump maintains Iran strikes
    caused 'total obliteration' Show all • Live Updates Live Updates
    Andrew Harnik/Getty Images Live Updates As a ceasefire appears to
    hold, Trump argues attack put Iran’s nuclear ambitions back decades
    but admits that intelligence report was ‘inconclusive’ CNN Exclusive
    Early US intel assessment suggests strikes only set Iran’s nuclear
    program back by months NATO summit yields a big win on defense
    spending for Trump but key questions over the alliance remain Video
    Hear how the White House responded to new report on US strikes on Iran
    3:47 Catch up on today's global news - Source: CNN Video Catch up on
    today’s news U.S. Air Force/Handout/Reuters Here’s what it can take
    for pilots to complete a 37-hour bombing mission Analysis Iran-Israel
    conflict: Is that it? Probably not. High tariffs give Trump less room
    for error in Iran Israel agreed to a ceasefire with Iran. Could Gaza
    be next? Ad Feedback More top stories Christian
    Monterrosa/Bloomberg/Getty Images Takeaways from New York City’s
    mayoral primary: Mamdani delivers a political earthquake Private
    mission lifts off with three astronauts who will be the first from
    their countries to visit the space station US Marine sentenced to 7
    years over sexual assault in Japan On the brink of retirement, this
    NFL player turned to psychedelics to help with his OCD Video At least
    49 people killed near aid sites in Gaza over 24-hour period 1:41 Why
    Kennedy’s overhaul of a key CDC committee could lead to ‘vaccine
    chaos’ in the US David W Cerny/Reuters Teenage sprint sensation Gout
    Gout breaks his own 200m Australian record in first race abroad Tom
    Booth/CNN The Great Barrier Reef is dying. Inside the lab trying to
    save it Edward Berthelot/Getty Images Luxury PJs and a more gentle
    approach: Here’s what we saw at the menswear shows in Milan • Video
    1:17 CCTV Video Why Japan has a rice crisis 1:17 Ad Feedback X/
    wangzhian8848 Video Truck with driver still inside dangles from bridge
    after landslide 0:41 Jun 24, 2025 Deaths from heart attacks are way
    down. Here’s what’s killing us instead Jun 25, 2025 Clipped From Video
    Disgruntled locals are ‘stacking’ Waymo cars so they can’t stir up
    noise at night Jun 25, 2025 CNN Video Amanpour asks Canadian PM: Is
    Trump still threatening annexation? 1:26 Jun 24, 2025 Ad Feedback
    Knight Frank Private Scottish island with ruined castle goes on sale
    for $7.5 million This animation, composed of 16 images acquired by
    NASA’s Terra satellite, shows Kati Thanda-Lake Eyre's evolution from
    April 29 to June 12. NASA Rare event breathes life back into
    Australia’s arid outback, attracting both animals and tourists • Video
    2:46 Getty Images/Getty Images North America/Getty Images • Video 2:46
    Video Costume designer reflects on why one character in ‘Sex and the
    City’ was ahead of her time 2:46 Ad Feedback Featured Sections Space
    and science Stephanie Wissel/Penn State Strange signals detected from
    Antarctic ice seem to defy laws of physics. Scientists are searching
    for an answer Scientists say a tiny brown moth navigates 600 miles
    using stars — just like humans and birds ‘Dragon Man’ DNA revelation
    puts a face to a mysterious group of ancient humans Global Travel
    Show all Mark Kerrison/In Pictures/Getty Images London has leaned into
    Jack the Ripper tourism. The locals don’t like it The beautiful but
    forgotten Bauhaus airport that’s an aviation time capsule London to
    Sweden for the day: These travelers are embracing extreme day trips
    The new nudity: A 21st-century guide to taking off your clothes Ad
    Feedback Global Business Frederic J. Brown/AFP/Getty Images Oil is
    falling so much it’s now cheaper than it was before Iran-Israel
    conflict After Iran uses missiles, US braces for cyberattacks High
    tariffs give Trump less room for error in Iran Why gasoline prices
    aren’t tumbling along with sinking oil Americans are feeling worse
    about the economy — especially Republicans Art and Style Victor
    Boyko/Getty Images A Dior debut and a fresh female perspective: What
    to watch at Paris Fashion Week Men’s Remember when Kurt Cobain spurned
    toxic masculinity in a dainty floral frock? Glastonbury Festival
    fashion history: Remember when Kate Moss wore rain boots? Milan
    Fashion Week Men’s begins. Here’s what to expect SPORT Show all
    Matthew Stockman/Getty Images Andy Murray confident Wimbledon statue
    will be improvement on Shanghai Masters’ infamous terracotta warrior
    CNN Exclusive Malala turns her fight for equality to women in sports
    Lyon relegated to French soccer’s second tier amid ongoing financial
    problems Aaron Rodgers on whether this upcoming NFL season is his
    last: ‘I’m pretty sure this is it’ Ad Feedback US Politics Anna
    Moneymaker/Getty Images Trump takes his go-it-alone approach to NATO
    summit after announcing Israel-Iran ceasefire Analysis Trump claims a
    forever peace in the land of forever war — but it could be fleeting
    Dramatic day of diplomacy culminates in Trump announcing Iran-Israel
    ceasefire Judge indefinitely blocks Trump’s proclamation suspending
    new Harvard international students Inside the campaign of Zohran
    Mamdani, the democratic socialist running for mayor of New York City
    health and wellness Ralf Geithe/iStockphoto/Getty Images BMI is B-A-D,
    a new study suggests. Here’s a better way to measure weight Deep belly
    fat triggers inflammation. Here’s how to reduce it Does face yoga
    actually work? Experts weigh in on its slimming, anti-aging effects
    The new coronavirus variant surging in China has arrived in the US.
    Here’s what to know Tech Stefani Reynolds/Bloomberg/Getty Images What
    is Perplexity, the AI startup said to be catching Meta and Apple’s
    attention Do you think AI is changing or threatening to take your job?
    Tell us about it here ‘She never sleeps’: This platform wants to be
    OnlyFans for the AI era Pope Leo calls for an ethical AI framework in
    a message to tech execs gathering at the Vatican Photos You Should See
    • Gallery Gallery The Legacy Collection/THA/Shutterstock Gallery In
    pictures: Behind the scenes of ‘Jaws’ • Gallery Gallery Lorenzo Poli
    Gallery Haunting image of an abandoned mining town wins Earth Photo
    prize • Gallery Gallery Dave Kotinsky/Getty Images Gallery People
    we’ve lost in 2025 • Gallery Gallery Royce L. Bair/DarkSky
    International Gallery Protecting our night skies from light pollution
    • Gallery Gallery Dendrinos/MOm Gallery Mediterranean monk seals: Back
    from the brink • Gallery Gallery Luiz Thiago de Jesus/AMLD Gallery
    Back from the brink: Golden lion tamarin Ad Feedback In Case You
    Missed It Guy Marineau/Conde Nast/Getty Images Towering heels, epic
    fall: Remember when Naomi Campbell turned a catwalk catastrophe into
    career gold? Evidence in Karen Read’s case led to ‘only one person,’
    prosecutor says in first statement since her acquittal Israel recovers
    bodies of three hostages – an IDF soldier and two civilians – from
    Gaza Suspect in Minnesota attacks drafted a ‘bailout plan’ for wife,
    according to court filing A clash of English and Brazilian teams in
    Philadelphia shows off exactly what FIFA wants from the Club World Cup
    US singer Chris Brown pleads not guilty to assault charge in UK court
    Wife of Colorado attack suspect says she and her 5 children are
    ‘suffering’ in ICE custody While North Korea denied Covid-19 cases,
    the virus was widespread and barely treated, report says World’s most
    liveable city for 2025 revealed Subscribe Sign in My Account Settings
    Newsletters Topics you follow Sign out Your CNN account Sign in to
    your CNN account Live TV Listen Watch World Africa Americas Asia
    Australia China Europe India Middle East United Kingdom US Politics
    Trump Facts First CNN Polls 2025 Elections Business Tech Media
    Calculators Videos Markets Pre-markets After-Hours Fear & Greed
    Investing Markets Now Nightcap Health Life, But Better Fitness Food
    Sleep Mindfulness Relationships Entertainment Movies Television
    Celebrity Tech Innovate Foreseeable Future Mission: Ahead Work
    Transformed Innovative Cities Style Arts Design Fashion Architecture
    Luxury Beauty Video Travel Destinations Food & Drink Stay News Videos
    Sports Football Tennis Golf Motorsport US Sports Olympics Climbing
    Esports Hockey Science Space Life Unearthed Climate Solutions Weather
    Weather Video Climate Ukraine-Russia War Israel-Hamas War Features As
    Equals Call to Earth Freedom Project Impact Your World Inside Africa
    CNN Heroes Watch Live TV CNN Fast Shows A-Z CNN10 CNN Max CNN TV
    Schedules Listen CNN 5 Things Chasing Life with Dr. Sanjay Gupta The
    Assignment with Audie Cornish One Thing Tug of War CNN Political
    Briefing The Axe Files All There Is with Anderson Cooper All CNN Audio
    podcasts Games Daily Crossword Jumble Crossword Sudoblock Sudoku 5
    Things Quiz About CNN Photos Investigations CNN Profiles CNN
    Leadership CNN Newsletters Work for CNN Watch Listen Live TV Follow
    CNN Subscribe Sign in My Account Settings Newsletters Topics you
    follow Sign out Your CNN account Sign in to your CNN account Terms of
    Use Privacy Policy Administrar cookies+ Ad Choices Accessibility & CC
    About Newsletters Transcripts © 2025 Cable News Network. A Warner
    Bros. Discovery Company. All Rights Reserved. CNN Sans ™ & © 2016
    Cable News Network.



```python
all_tools = [tool_send_email] + tools
```


```python
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
llm_with_tools = llm.bind_tools(all_tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}
```


```python
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=all_tools))
graph_builder.add_conditional_edges( "chatbot", tools_condition, "tools")
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
display(Image(graph.get_graph().draw_mermaid_png()))
```


    
![png](23_LangGraph%20Project%20%28i%29_files/23_LangGraph%20Project%20%28i%29_17_0.png)
    



```python
# Esto evita el error: There is no current event loop in thread 'MainThread'
import uvicorn

uvicorn.config.LOOP_SETUPS = {
    "none": None,
    "auto": "uvicorn.loops.asyncio:asyncio_setup",
    "asyncio": "uvicorn.loops.asyncio:asyncio_setup",
    "uvloop": "uvicorn.loops.uvloop:uvloop_setup",
}
```


```python
config = {"configurable": {"thread_id": "1"}}

async def chat(user_input: str, history):
    result = await graph.ainvoke({"messages": [{"role": "user", "content": user_input}]}, config=config)
    return result["messages"][-1].content


gr.ChatInterface(chat, type="messages").launch()
```

    * Running on local URL:  http://127.0.0.1:7863
    * To create a public link, set `share=True` in `launch()`.



<div><iframe src="http://127.0.0.1:7863/" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>





    


