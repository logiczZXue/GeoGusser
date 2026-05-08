#!/usr/bin/env python3
"""
Street View API Wrapper Module

Provides methods for starting/stopping the service, modifying the frontend configuration, and calling various API endpoints.
"""

import os
import re
import time
import subprocess
import logging
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

class StreetViewAPI:
    """
    This class encapsulates operations for interacting with the Street View page.
    
    Provided API endpoints include:
      1. get_status  : Returns a screenshot (PNG format) of the current page
      2. look_left   : Simulates a left rotation
      3. look_right  : Simulates a right rotation
      4. look_up     : Simulates an upward tilt
      5. look_down   : Simulates a downward tilt
      6. forward     : Simulates moving forward
      7. backward    : Simulates moving backward
      8. turn_left   : Simulates turning left
      9. turn_right  : Simulates turning right
      10. zoom_in    : Simulates zooming in (mouse wheel scroll up)
      11. zoom_out   : Simulates zooming out (mouse wheel scroll down)
    
    Usage:
      1. Provide google_map_api_key, app_port, api_port, lat, and lng upon initialization.
         The module automatically updates configuration files (public/index.html, app.js, and api.js).
      2. Call start_servers() to start the services, or use the "with" statement for automatic management.
      3. Use the API methods to control the Street View; get_status returns binary image data.
    """
    def __init__(self, google_map_api_key, app_port, api_port, lat, lng):
        self.google_map_api_key = google_map_api_key
        self.app_port = app_port
        self.api_port = api_port
        self.lat = lat
        self.lng = lng
        self.app_process = None
        self.api_process = None
        self.base_api_url = f"http://localhost:{self.api_port}"

    def update_index_html(self):
        """
        Update the Google Maps API key and initial coordinates in public/index.html.
        """
        index_path = os.path.join("public", "index.html")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"File not found: {index_path}")
        
        with open(index_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Replace the Google Maps API key
        new_content = re.sub(
            r"(https:\/\/maps\.googleapis\.com\/maps\/api\/js\?key=)[^\"&\s]+",
            r"\1" + self.google_map_api_key,
            content
        )

        # Replace the initial coordinates; matching a pattern like: const fenway = { lat: 42.345573, lng: -71.098326 };
        new_content = re.sub(
            r"(const\s+fenway\s*=\s*{\s*lat:\s*)[-\d.]+(,\s*lng:\s*)[-\d.]+(\s*};)",
            lambda m: m.group(1) + str(self.lat) + m.group(2) + str(self.lng) + m.group(3),
            new_content
        )

        with open(index_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        logging.info("Updated public/index.html with new Google Maps API key and initial coordinates")

    def update_js_ports(self):
        """
        Update port settings in app.js and api.js.
        """
        # Update app.js
        app_js_path = "app.js"
        if not os.path.exists(app_js_path):
            raise FileNotFoundError(f"File not found: {app_js_path}")
        with open(app_js_path, "r", encoding="utf-8") as f:
            app_js_content = f.read()
        app_js_modified = re.sub(
            r"(const\s+PORT\s*=\s*)\d+;",
            lambda m: m.group(1) + str(self.app_port) + ";",
            app_js_content
        )
        with open(app_js_path, "w", encoding="utf-8") as f:
            f.write(app_js_modified)
        logging.info(f"Updated PORT in app.js to {self.app_port}")

        # Update api.js
        api_js_path = "api.js"
        if not os.path.exists(api_js_path):
            raise FileNotFoundError(f"File not found: {api_js_path}")
        with open(api_js_path, "r", encoding="utf-8") as f:
            api_js_content = f.read()
        api_js_modified = re.sub(
            r"(const\s+API_PORT\s*=\s*)\d+;",
            lambda m: m.group(1) + str(self.api_port) + ";",
            api_js_content
        )
        with open(api_js_path, "w", encoding="utf-8") as f:
            f.write(api_js_modified)
        logging.info(f"Updated API_PORT in api.js to {self.api_port}")

    def start_servers(self):
        """
        Start the app.js and api.js services, and update the related configuration files.
        """
        self.update_index_html()
        self.update_js_ports()

        # Start services using Node.js
        self.app_process = subprocess.Popen(["node", "app.js"])
        logging.info(f"Started app.js service on port {self.app_port}")
        self.api_process = subprocess.Popen(["node", "api.js"])
        logging.info(f"Started api.js service on port {self.api_port}")

        # Simple delay to ensure services are fully up (consider using a polling mechanism)
        time.sleep(10)

    def restore_privacy_info(self):
        """
        Restore privacy-related configuration in public/index.html, app.js, and api.js:
          - In index.html, revert the Google Maps API key to GOOGLE_MAP_API_KEY and initial coordinates to (0,0)
          - In app.js, revert PORT to 1
          - In api.js, revert API_PORT to 2
        """
        # Restore index.html
        index_path = os.path.join("public", "index.html")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                content = f.read()
            new_content = re.sub(
                r"(https:\/\/maps\.googleapis\.com\/maps\/api\/js\?key=)[^\"&\s]+",
                r"\1GOOGLE_MAP_API_KEY",
                content
            )
            new_content = re.sub(
                r"(const\s+fenway\s*=\s*{\s*lat:\s*)[-\d.]+(,\s*lng:\s*)[-\d.]+(\s*};)",
                lambda m: m.group(1) + "0" + m.group(2) + "0" + m.group(3),
                new_content
            )
            with open(index_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            logging.info("Restored privacy settings in public/index.html")
        else:
            logging.warning(f"File not found: {index_path}. Skipping privacy restore for index.html")

        # Restore app.js
        app_js_path = "app.js"
        if os.path.exists(app_js_path):
            with open(app_js_path, "r", encoding="utf-8") as f:
                app_js_content = f.read()
            app_js_modified = re.sub(
                r"(const\s+PORT\s*=\s*)\d+;",
                lambda m: m.group(1) + "1;",
                app_js_content
            )
            with open(app_js_path, "w", encoding="utf-8") as f:
                f.write(app_js_modified)
            logging.info("Restored PORT to 1 in app.js")
        else:
            logging.warning(f"File not found: {app_js_path}. Skipping privacy restore for app.js")

        # Restore api.js
        api_js_path = "api.js"
        if os.path.exists(api_js_path):
            with open(api_js_path, "r", encoding="utf-8") as f:
                api_js_content = f.read()
            api_js_modified = re.sub(
                r"(const\s+API_PORT\s*=\s*)\d+;",
                lambda m: m.group(1) + "2;",
                api_js_content
            )
            with open(api_js_path, "w", encoding="utf-8") as f:
                f.write(api_js_modified)
            logging.info("Restored API_PORT to 2 in api.js")
        else:
            logging.warning(f"File not found: {api_js_path}. Skipping privacy restore for api.js")

    def stop_servers(self):
        """
        Stop the running services and restore the privacy configurations.
        """
        if self.app_process:
            self.app_process.terminate()
            self.app_process.wait()
            logging.info("Stopped app.js service")
        if self.api_process:
            self.api_process.terminate()
            self.api_process.wait()
            logging.info("Stopped api.js service")
        self.restore_privacy_info()

    def _call_api(self, endpoint):
        """
        Internal method - Sends a GET request to the specified API endpoint.
        
        :param endpoint: The URL path of the API endpoint (e.g., '/api/look_left')
        :return: requests.Response object
        """
        url = f"{self.base_api_url}{endpoint}"
        try:
            response = requests.get(url)
            return response
        except Exception as e:
            raise Exception(f"Failed to call API {endpoint}: {e}")

    def get_status(self, save_path=None):
        """
        Call the /api/get_status endpoint to capture a screenshot of the current Street View (PNG format).
        
        :param save_path: Optional file path to save the image; if None, the image is not saved.
        :return: Binary image data.
        """
        response = self._call_api("/api/get_status")
        if response.status_code == 200:
            img_data = response.content
            if save_path:
                try:
                    with open(save_path, "wb") as f:
                        f.write(img_data)
                    logging.info(f"Image saved to {save_path}")
                except Exception as e:
                    logging.error(f"Error saving image: {e}")
            return img_data
        else:
            raise Exception(f"get_status API call failed with status code: {response.status_code}")

    def look_left(self):
        return self._call_api("/api/look_left").text

    def look_right(self):
        return self._call_api("/api/look_right").text

    def look_up(self):
        return self._call_api("/api/look_up").text

    def look_down(self):
        return self._call_api("/api/look_down").text

    def forward(self):
        return self._call_api("/api/forward").text

    def backward(self):
        return self._call_api("/api/backward").text

    def turn_left(self):
        return self._call_api("/api/turn_left").text

    def turn_right(self):
        return self._call_api("/api/turn_right").text

    def zoom_in(self):
        return self._call_api("/api/zoom_in").text

    def zoom_out(self):
        return self._call_api("/api/zoom_out").text

    def __enter__(self):
        self.start_servers()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_servers()


if __name__ == "__main__":
    # Usage example: Ensure that the "public" folder, app.js, and api.js exist in the project directory,
    # and that Node.js along with required dependencies (e.g., express, puppeteer) are installed.
    #
    # Replace the following google_map_api_key with a valid API key

    google_map_api_key = "xxx"
    app_port = 6657    # It is recommended to use ports higher than 1024
    api_port = 5200
    lat = 42.345573
    lng = -71.098326

    with StreetViewAPI(google_map_api_key, app_port, api_port, lat, lng) as sv_api:
        logging.info("look_left API call response: " + sv_api.look_left())
        logging.info("look_right API call response: " + sv_api.look_right())
        logging.info("look_up API call response: " + sv_api.look_up())
        logging.info("look_down API call response: " + sv_api.look_down())
        logging.info("forward API call response: " + sv_api.forward())
        logging.info("backward API call response: " + sv_api.backward())
        logging.info("turn_left API call response: " + sv_api.turn_left())
        logging.info("turn_right API call response: " + sv_api.turn_right())
        logging.info("zoom_in API call response: " + sv_api.zoom_in())
        logging.info("zoom_out API call response: " + sv_api.zoom_out())

        _ = sv_api.get_status(save_path="current_status.png")
        logging.info("Screenshot saved as current_status.png")
        time.sleep(5) 