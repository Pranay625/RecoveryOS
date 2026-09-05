from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str
    GEMINI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    RAZORPAY_KEY_ID: str = ""
    RAZORPAY_KEY_SECRET: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
