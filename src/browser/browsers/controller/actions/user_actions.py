import logging
import aiohttp
from browser_use.browser.context import BrowserContext

logger = logging.getLogger(__name__)

class UserActionsMixin:
    """
    Mixin for user-defined actions.
    Add your custom tools here to extend the agent's capabilities without modifying core files.
    """
    def _register_user_actions(self):
        
        @self.registry.action("Example Custom Tool: Print a greeting")
        async def my_custom_tool(browser: BrowserContext, name: str):
            """
            An example tool that logs a greeting.
            Args:
                browser: The browser context.
                name: The name to greet.
            """
            message = f"Hello, {name}! This is a custom tool executing."
            logger.info(message)
            return message

        @self.registry.action("Get weather for a city using Open-Meteo API")
        async def get_weather(browser: BrowserContext, city: str):
            """
            Fetches current weather for a city.
            Args:
                city: Name of the city (e.g., 'London', 'New York').
            """
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
            
            try:
                async with aiohttp.ClientSession() as session:
                    # 1. Geocode the city name
                    async with session.get(geo_url) as geo_resp:
                        if geo_resp.status != 200:
                            return f"Geocoding failed for {city}"
                        geo_data = await geo_resp.json()
                        
                    if not geo_data.get("results"):
                        return f"City '{city}' not found."
                        
                    lat = geo_data["results"][0]["latitude"]
                    lon = geo_data["results"][0]["longitude"]
                    
                    # 2. Fetch Weather
                    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
                    async with session.get(weather_url) as w_resp:
                        if w_resp.status != 200:
                            return "Weather API failed."
                        w_data = await w_resp.json()
                        
                    current = w_data.get("current_weather", {})
                    return f"Weather in {city}: {current.get('temperature')}°C, Wind: {current.get('windspeed')} km/h"
                    
            except Exception as e:
                logger.error(f"API Error: {e}")
                return f"Error fetching weather: {e}"
