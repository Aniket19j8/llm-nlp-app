from app.settings import settings
def mask(s): 
    return (s[:4] + "..." + s[-4:]) if s else None
print("API key:", mask(settings.OPENAI_API_KEY))
print("BASE URL:", settings.OPENAI_BASE_URL)
print("MODEL   :", settings.OPENAI_MODEL)
