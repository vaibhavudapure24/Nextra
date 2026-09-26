"""
Zero-configuration FastAPI Backend Entry Point for Vercel & Local Execution.
Exports the FastAPI app instance from api.main.
"""
from api.main import app

if __name__ == "__main__":
    import uvicorn
    from utils.config_loader import load_config
    cfg = load_config()
    uvicorn.run("main:app", host=cfg.api.host, port=cfg.api.port, reload=True)
