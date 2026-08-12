import contextvars

# OpenTelemetry handles its own trace_id and span_id in the span context, 
# but we maintain high-level business correlation IDs here so that structlog 
# and other components can easily inject them without calling OTEL APIs everywhere.

# Define context variables
execution_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("execution_id", default=None)
conversation_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("conversation_id", default=None)
user_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("user_id", default=None)

def get_execution_id() -> str | None:
    return execution_id_var.get()

def set_execution_id(execution_id: str):
    execution_id_var.set(execution_id)

def get_conversation_id() -> str | None:
    return conversation_id_var.get()

def set_conversation_id(conversation_id: str):
    conversation_id_var.set(conversation_id)
    
def get_user_id() -> str | None:
    return user_id_var.get()

def set_user_id(user_id: str):
    user_id_var.set(user_id)
