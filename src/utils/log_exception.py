import traceback

def log_exception(e):
    return f"[{type(e).__name__}] {e}\n{traceback.format_exc()}"
