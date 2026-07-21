from __future__ import annotations
import json
import operator
from typing import Annotated, Sequence, TypedDict, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.tools import tool
from langgraph.graph import StateGraph


MODEL_NAME = "deepseek-v2:16b"

@tool
def investigate_transaction(account_id: str = None, customer_id: str = None, alert_type: str = None, **kwargs) -> str:
    """Investigate suspicious transactions, fraud alerts, or security concerns."""
    print(f"[TOOL] investigate_transaction(account_id={account_id}, alert_type={alert_type}, kwargs={kwargs})")
    return "investigation_initiated"

@tool
def freeze_account(account_id: str, reason: str, freeze_type: str = "immediate", customer_request: bool = True) -> str:
    """Freeze account to prevent unauthorized access or transactions."""
    print(f"[TOOL] freeze_account(account_id={account_id}, reason={reason}, freeze_type={freeze_type})")
    return "account_frozen"

@tool
def process_loan_application(customer_id: str, loan_type: str, loan_amount: str = None, **kwargs) -> str:
    """Process loan applications including personal, business, mortgage, and auto loans."""
    print(f"[TOOL] process_loan_application(customer_id={customer_id}, loan_type={loan_type}, amount={loan_amount})")
    return "application_submitted"

@tool
def resolve_dispute(account_id: str = None, customer_id: str = None, dispute_type: str = None, **kwargs) -> str:
    """Handle disputes including unauthorized charges, fees, and credit report errors."""
    print(f"[TOOL] resolve_dispute(account_id={account_id}, dispute_type={dispute_type}, kwargs={kwargs})")
    return "dispute_filed"

@tool
def rebalance_portfolio(customer_id: str, **kwargs) -> str:
    """Manage investment portfolios, retirement planning, and asset allocation."""
    print(f"[TOOL] rebalance_portfolio(customer_id={customer_id}, kwargs={kwargs})")
    return "portfolio_updated"

@tool
def increase_credit_limit(account_id: str, current_limit: str, requested_limit: str, **kwargs) -> str:
    """Process credit limit increase requests."""
    print(f"[TOOL] increase_credit_limit(account_id={account_id}, current={current_limit}, requested={requested_limit})")
    return "credit_limit_updated"

@tool
def verify_documents(customer_id: str, **kwargs) -> str:
    """Verify customer documents for various banking services."""
    print(f"[TOOL] verify_documents(customer_id={customer_id}, kwargs={kwargs})")
    return "documents_verified"

@tool
def update_account(account_id: str = None, customer_id: str = None, **kwargs) -> str:
    """Update account information, add joint holders, close accounts, etc."""
    print(f"[TOOL] update_account(account_id={account_id}, customer_id={customer_id}, kwargs={kwargs})")
    return "account_updated"

@tool
def process_transaction(customer_id: str, transaction_type: str, **kwargs) -> str:
    """Process various transactions like currency exchange, transfers, etc."""
    print(f"[TOOL] process_transaction(customer_id={customer_id}, type={transaction_type}, kwargs={kwargs})")
    return "transaction_processed"

@tool
def send_customer_response(customer_id: str, message: str) -> str:
    """Send a response message to the customer."""
    print(f"[TOOL] send_customer_response → {message}")
    return "message_sent"

TOOLS = [
    investigate_transaction, freeze_account, process_loan_application, resolve_dispute,
    rebalance_portfolio, increase_credit_limit, verify_documents, update_account,
    process_transaction, send_customer_response
]

llm = ChatOllama(model=MODEL_NAME, temperature=0.0, callbacks=[StreamingStdOutCallbackHandler()],  
    verbose=True).bind_tools(TOOLS)

class AgentState(TypedDict):
    account: Optional[dict]  # Customer account information
    messages: Annotated[Sequence[BaseMessage], operator.add]

def call_model(state: AgentState):
    history = state["messages"]
    
    # Handle missing or incomplete account data gracefully
    account = state.get("account", {})
    if not account:
        account = {"account_id": "UNKNOWN", "customer_id": "UNKNOWN", "status": "active"}
    
    account_json = json.dumps(account, ensure_ascii=False)
    system_prompt = (
        "You are a professional financial services agent specializing in banking, fraud prevention, loans, and investments.\n"
        "When you assist customers, you should:\n"
        "  1) Analyze their request and call the appropriate business tool\n"
        "  2) Call send_customer_response with a helpful confirmation message\n"
        "Always prioritize security and compliance with banking regulations.\n\n"
        f"ACCOUNT: {account_json}"
    )

    full = [SystemMessage(content=system_prompt)] + history

    first: ToolMessage | BaseMessage = llm.invoke(full)
    messages = [first]

    if getattr(first, "tool_calls", None):
        for tc in first.tool_calls:
            print(first)
            print(tc['name'])
            fn = next(t for t in TOOLS if t.name == tc['name'])
            out = fn.invoke(tc["args"])
            messages.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))

        second = llm.invoke(full + messages)
        messages.append(second)

    return {"messages": messages}

def construct_graph():
    g = StateGraph(AgentState)
    g.add_node("assistant", call_model)
    g.set_entry_point("assistant")
    return g.compile()

graph = construct_graph()

if __name__ == "__main__":
    example = {"account_id": "ACC123456", "customer_id": "CUST789", "balance": 5000.0}
    convo = [HumanMessage(content="I think there's fraud on my account. I see a $2,500 charge in Miami but I'm in New York.")]
    result = graph.invoke({"account": example, "messages": convo})
    for m in result["messages"]:
        print(f"{m.type}: {m.content}") 
