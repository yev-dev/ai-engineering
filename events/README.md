1. Create langchain processor that has registered react agents for handling user request waiting for user prompt
2. Create handler based on langchain framework that would action on human in a loop event
3. Have another react agents to handle on user prompt. One agent would should manage car booking and another air ticket and another agent for hotel reservation. If car booking is selected send another event to select a car waiting for user reponse to confirm car type.
4. Have registered tools
5. Do not hard code workflow, everythong should LLM driven with ReACT agent
6. Have agents definition 
7. Use local ollama and litellm framework and gemma4:e4b model 
8. Create cli.py to handle 
9. create processor.py, cli.py, agents.py, handler.py