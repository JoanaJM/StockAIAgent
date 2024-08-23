import json
import os
from datetime import datetime

import yfinance as yf

from crewai import Agent, Task, Process, Crew

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults

import streamlit as st


#Yahoo finance Tool
def fetch_stock_price(ticket):
    stock = yf.download("AAPL", start="2023-08-08", end="2024-08-08")
    return stock

yahoo_finance_tool = Tool(
    name = "Yahoo Finance Tool",
    description = "Fetches stock prices for {ticket} from the last year about a specific company from Yahoo Finance API",
    func= lambda ticket: fetch_stock_price(ticket)
)

#Agent

os.environ['OPENAI_KEY'] = st.secrets('OPEANAI_KEY')

llm = ChatOpenAI(model="gpt-3.5-turbo")

#AAPL é o nome da ação da Apple

stockPriceAnalyst = Agent(
    role='Senior stock price analyst',
    goal='Find the {ticket} stock price and analyses trends',
    backstory="""You are highly experienced in analyzing the price of an specific stock
    and making predictions about its future price.""",
    verbose=True,
    llm=llm,
    max_iter=5,
    memory=True,
    allow_delegation=False,
    tools=[yahoo_finance_tool]
)


getStockPrice = Task(
    description='Analyse the stock {ticket} price history and creat a trend analyses of up, down or sideways',
    expected_output="""Specify the current trend stock price - up, down or sideways.
    eg. stock = 'APPL, price UP' """,
    agent=stockPriceAnalyst
)



search_tool = DuckDuckGoSearchResults(backend='news', num_results=10)


newsAnalyst = Agent(
    role= "Stock news analyst",
    goal= """Create a shor summary of the market news related to the stock {ticket} company. 
    Specify the current trend - up, down or sideways with the new context.
    For each request stock asset, specify a number between 0 and 100, where 0 is extream fear and 100 is extream greed.""",
    backstory="""You're highly experienced in analyzing the market trends and news and hjave tracked assets for more then 10 years.
    You're also master level analyst in the tradicional markets and have deep undestanding of human psychology.
    You understand news, their tittles and information, but you look at those with a health dose of skepticism.
    You consider also the source of the news articles.""",
    verbose=True,
    llm= llm,
    max_iter=5,
    memory=True,
    allow_delegation=False,
    tools=[search_tool]
)


get_news = Task(
    description=f""" Take the stock and alwasys include BTC to it (if not requested).
    Use the search tool to search each one individually.
    The current date is {datetime.now()}.
    Compose the results into a helpfull report """,
    expected_output=""" A summary of the overall market and one sentance sumary for each request asset.
    Includse a fear/greet score for each asset based on the news. Use format:
    <STOCK ASSET>
    <SUMMARY BASED ON NEWS>
    <TREND PREDICTION>
    <FEAR/GREED SCORE>
    """,
    agent= newsAnalyst
)


stockAnalystWrite = Agent(
    role="Senior Stock Analyst Writer",
    goal= """Analyse the trends price and news and write an insighfull compelling and informative 3 paragraph long newsletter based on the sotck report and price trend.""",
    backstory="""You're widely accepted as the best stock analyst in the market. You understand complex concepts and create compelling stories and narratives that resoneate with wider audiences.
    You undestand macro factors and combine multiple theories - eg. cycle theory and fundamental analyses.
    You're able to hold multiple opinions when analyzing anything.""",
    verbose=True,
    llm=llm,
    max_iter=10,
    memory=True,
    allow_delegation=True
)


writeAnalyses = Task(
    description="""Use the stock price trend and the stock news report to create an analysis and write the newsletter about
    the {ticket} company that is brief and highlights the most important points.
    Focus on the stock price trend, news and fear/greed score. What are the near furure considerations?
    Include the previous analysis of stock trend and news summary.""",
    expected_output="""An eloquent 3 paragraph newsletter formated as markdown in an easy readable manner. It should contain:
    - 3 bullerts executive summary
    - Introduction - set the overall picture and spike up the interest
    - main part provides the meat of the analysis including the news summary and fear/greed scores
    - summary - key facts and concrete future trend prediction - up, down or sideways.""",
    agent= stockAnalystWrite,
    context= [getStockPrice, get_news]
)



crew = Crew(
    agents= [stockPriceAnalyst, newsAnalyst, stockAnalystWrite],
    tasks= [getStockPrice, get_news, writeAnalyses],
    verbose = True,
    process=Process.hierarchical, 
    full_output=True,
    share_crew=False,
    manager_llm=llm,
    max_iter=15
)


#Streamlit App
with st.sidebar:
    st.header('Enter the Stock to Research')

    with st.form(key='research_form'):
        topic = st.text_input('Select the ticket')
        submit_button = st.form_submit_button(label='Run Search')

if submit_button:
    if not topic:
        st.error('Please fill the ticket field')
    else:
        results = crew.kickoff(inputs={'ticket': topic})
        st.subheader('Results of yout research:')
        st.write(results['final_output'])
