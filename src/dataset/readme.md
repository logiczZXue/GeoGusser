# StreetViewAPI - README

## Project Overview

StreetViewAPI is an application built using Python, Node.js, Express, and Puppeteer to simulate various operations on Google Street View. The project consists of three main components:

  - Frontend:
      • public/index.html: Loads Google Street View using the Google Maps JavaScript API and sets the initial coordinates.
      • app.js: An Express-based static server that serves the files in the public directory.
  
  - Backend API:
      • api.js: Provides 11 RESTful API endpoints using Puppeteer to simulate keyboard and mouse events (e.g., rotating the view, moving, zooming).
  
  - Python Wrapper:
      • street_view_api.py: Encapsulates API calls in the StreetViewAPI class. It automatically updates configuration files (Google Maps API key, initial coordinates, port settings), manages service startup and shutdown, and restores default privacy-related settings upon termination.

## API Reference

### 2.1. StreetViewAPI.get_status(save_path=None) → bytes

**Description:**
Captures a screenshot of the current Street View page.

**Parameters:**
    save_path (str, optional): The file path where the screenshot will be saved. If set to None (default), the image is not saved to disk.

**Returns:**
    bytes: The PNG image in binary format.

**Examples:**
    With save_path specified:

```python
img_data = sv_api.get_status(save_path="current_status.png")
print("Screenshot saved as current_status.png")
```

​    With save_path omitted (save_path=None):

```python
img_data = sv_api.get_status()
# Process the returned image data as needed
```


### 2.2. StreetViewAPI.look_left() → str
**Description:**
Simulates a left rotation of the view by emulating the press of the "j" key.

**Returns:**
    str: A textual confirmation of the left rotation action.

**Example:**

```python
result = sv_api.look_left()
print("Response:", result)
```


### 2.3. StreetViewAPI.look_right() → str
**Description:**
Simulates a right rotation of the view by emulating the press of the "l" key.

**Returns:**
    str: A textual confirmation of the right rotation action.

**Example:**

```python
result = sv_api.look_right()
print("Response:", result)
```


### 2.4. StreetViewAPI.look_up() → str
**Description:**
Simulates tilting the view upward by emulating the press of the "i" key.

**Returns:**
    str: A textual confirmation of the upward tilt action.

**Example:**

```python
result = sv_api.look_up()
print("Response:", result)
```


### 2.5. StreetViewAPI.look_down() → str
**Description:**
Simulates tilting the view downward by emulating the press of the "k" key.

**Returns:**
    str: A textual confirmation of the downward tilt action.

**Example:**

```python
result = sv_api.look_down()
print("Response:", result)
```


### 2.6. StreetViewAPI.forward() → str
**Description:**
Simulates moving forward by emulating the press of the "w" key.

**Returns:**
    str: A textual confirmation of the forward movement.

**Example:**

```python
result = sv_api.forward()
print("Response:", result)
```


### 2.7. StreetViewAPI.backward() → str
**Description:**
Simulates moving backward by emulating the press of the "s" key.

**Returns:**
    str: A textual confirmation of the backward movement.

**Example:**

```python
result = sv_api.backward()
print("Response:", result)
```


### 2.8. StreetViewAPI.turn_left() → str
**Description:**
Simulates turning left by emulating the press of the "a" key.

**Returns:**
    str: A textual confirmation of the left turn action.

**Example:**

```python
result = sv_api.turn_left()
print("Response:", result)
```


### 2.9. StreetViewAPI.turn_right() → str
**Description:**
Simulates turning right by emulating the press of the "d" key.

**Returns:**
    str: A textual confirmation of the right turn action.

**Example:**

```python
result = sv_api.turn_right()
print("Response:", result)	
```


### 2.10. StreetViewAPI.zoom_in() → str
**Description:**
Simulates zooming in. This method first clicks at the center of the page and then emulates a mouse wheel scroll up.

**Returns:**
    str: A textual confirmation of the zoom in action.

**Example:**

```python
result = sv_api.zoom_in()
print("Response:", result)
```


### 2.11. StreetViewAPI.zoom_out() → str
**Description:**
Simulates zooming out. This method first clicks at the center of the page and then emulates a mouse wheel scroll down.

**Returns:**
    str: A textual confirmation of the zoom out action.

**Example:**

```python
result = sv_api.zoom_out()
print("Response:", result)
```


## Service Usage Examples

### 3.1. Using the "with" Statement for Automatic Service Management

**Description:**
The "with" statement automatically starts the services upon entry and stops them (restoring privacy settings) upon exit.

**Example:**

```python
with StreetViewAPI(google_map_api_key, app_port, api_port, lat, lng) as sv_api:
    print("Response from look_left:", sv_api.look_left())
    print("Response from look_right:", sv_api.look_right())
    print("Response from look_up:", sv_api.look_up())
    print("Response from look_down:", sv_api.look_down())
    print("Response from forward:", sv_api.forward())
    print("Response from backward:", sv_api.backward())
    print("Response from turn_left:", sv_api.turn_left())
    print("Response from turn_right:", sv_api.turn_right())
    print("Response from zoom_in:", sv_api.zoom_in())
    print("Response from zoom_out:", sv_api.zoom_out())
    img_data = sv_api.get_status(save_path="current_status.png")
    print("Screenshot saved as current_status.png")
    time.sleep(5)
```

### 3.2. Manual Service Control (start_servers/stop_servers)
**Description:**
Manually control the service lifecycle by calling start_servers() to start and stop_servers() to stop the services.

**Example:**

```python
sv_api = StreetViewAPI(google_map_api_key, app_port, api_port, lat, lng)

sv_api.start_servers()
print("Response from forward:", sv_api.forward())

# Perform additional operations as needed...

sv_api.stop_servers()
```