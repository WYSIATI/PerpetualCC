# What is this
This project is aimed to run a coding agent 24/7 until a project is finally finished, starting with Claude Code as MVP

# Pain points
- In Claude Code or other coding agent tasks, I have to wait until a sesssion finish one task, sometimes I got lost with other things, and forget to come back to continue coding.
- When token limit reset, I fogot to come back immediately to work
- For multiple sessions in Cluade Code, it is hard to switch in between

# Key features
1. Read requirements from user direct input, or a file to understand what are required for the project.
2. Detect when the Claude Code tokens run out and limit reset time, resume prompting Claude Code when the limit resets.
3. Know which session in Claude Code to prompt. This is like a master agent, it needs to be able to connect to Claude Code somehow, and send prompts to Claude Code
4. Know when and how to answer Claude Code's question during execution
5. Able to integrate or inject into Claude Code sessions, this master agent could run multiple sessions, and know which session is paused and waiting for user input (now the user is master agent)
6. During daytime, a real person interferes, the master agent could forward the interaction message to human, so that human is able to answer questions or interact (clicks or other operations) with multiple sessions in Claude Code within one interface. The interface could be a local web interface or a command line interface



# 3 types of brains
1. No AI, simply execute the requirements one by one
2. Integrate Gemini API, so that it understads how to answer questions from Claude Code, and knows what is the next step
3. Use local LLM, probably DeepSeek or other openn sources model could be used.

# Tech stack
Not limited

# User onbaording
The user setup should be very user friendly, if user installs local llm or integrates with Gemini API, it should be very easy. If it is an app, design it for Mac OS only.
