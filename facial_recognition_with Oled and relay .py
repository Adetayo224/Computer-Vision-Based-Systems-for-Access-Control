import face_recognition
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import pickle
from gpiozero import LED
from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import sh1106, ssd1306
from PIL import Image, ImageDraw, ImageFont
import threading

# Load pre-trained face encodings
print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Initialize the camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (1920, 1080)}))
picam2.start()

# Initialize GPIO
output = LED(14)

# Initialize OLED Display
print("[INFO] Initializing OLED display...")
device = None
# Try both possible addresses for your display (0x78=0x3C, 0x7A=0x3D in 7-bit)
addresses_to_try = [0x3C, 0x3D]  # Your 0x78 and 0x7A in 7-bit format

for addr in addresses_to_try:
    try:
        print(f"[INFO] Trying OLED at I2C address 0x{addr:02X} (8-bit: 0x{addr*2:02X})")
        serial = i2c(port=1, address=addr)
        # Try SH1106 first (most common for 1.3" displays)
        try:
            device = sh1106(serial)
            print(f"[INFO] OLED (SH1106) initialized at address 0x{addr:02X}")
            break
        except:
            # If SH1306 fails, try SSD1306
            device = ssd1306(serial)
            print(f"[INFO] OLED (SSD1306) initialized at address 0x{addr:02X}")
            break
    except Exception as e:
        print(f"[INFO] Failed at address 0x{addr:02X}: {e}")
        continue

if device is None:
    print("[ERROR] Could not initialize OLED display at any address")
    print("[INFO] System will continue without OLED display")

# Try to load a font, fallback to default if not available
try:
    font_small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 10)
    font_medium = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
    font_large = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 16)
except:
    font_small = ImageFont.load_default()
    font_medium = ImageFont.load_default()
    font_large = ImageFont.load_default()

# Initialize our variables
cv_scaler = 4 # this has to be a whole number

face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0

# Delay variables for GPIO control
relay_trigger_time = 0
relay_delay = 5.0  # 5 seconds delay
relay_active = False

# SECURITY ENHANCEMENT VARIABLES
CONFIDENCE_THRESHOLD = 0.45  # Lower distance = higher confidence (0.6 is default, 0.45 is stricter)
REQUIRED_CONFIRMATIONS = 3    # Number of consecutive confirmations needed
CONFIRMATION_WINDOW = 2.0     # Time window for confirmations (seconds)
MIN_FACE_AREA = 8000         # Minimum face area in pixels (prevents tiny/distant faces)

# Confirmation tracking
confirmation_data = {
    'count': 0,
    'name': None,
    'start_time': 0,
    'confidences': [],
    'face_areas': []
}

# Display variables
last_detected_name = "None"
detection_time = ""
system_status = "Ready"
current_confidence = 0.0
light_warning = False

# List of names that will trigger the GPIO pin
authorized_names = ["Ayoola", "Adetayo", "bob"]  # Replace with names you wish to authorise THIS IS CASE-SENSITIVE

def calculate_brightness(frame):
    """Calculate average brightness of the frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def calculate_face_area(face_location):
    """Calculate face area in pixels"""
    top, right, bottom, left = face_location
    return (right - left) * (bottom - top)

def is_good_quality_detection(face_distance, face_area, frame_brightness):
    """Check if detection meets quality requirements"""
    confidence_good = face_distance <= CONFIDENCE_THRESHOLD
    size_good = face_area >= MIN_FACE_AREA
    light_adequate = frame_brightness >= 50  # Minimum brightness threshold
    
    return confidence_good and size_good and light_adequate, {
        'confidence': confidence_good,
        'size': size_good,
        'lighting': light_adequate,
        'brightness': frame_brightness
    }

def update_confirmation(name, confidence, face_area):
    """Update confirmation tracking"""
    global confirmation_data
    current_time = time.time()
    
    # If it's a new person or too much time has passed, reset
    if (confirmation_data['name'] != name or 
        current_time - confirmation_data['start_time'] > CONFIRMATION_WINDOW):
        confirmation_data = {
            'count': 1,
            'name': name,
            'start_time': current_time,
            'confidences': [confidence],
            'face_areas': [face_area]
        }
    else:
        # Same person within time window, increment
        confirmation_data['count'] += 1
        confirmation_data['confidences'].append(confidence)
        confirmation_data['face_areas'].append(face_area)
    
    return confirmation_data['count'] >= REQUIRED_CONFIRMATIONS

def update_oled_display():
    """Update OLED display with current status"""
    if device is None:
        return
    
    try:
        with canvas(device) as draw:
            # Title
            draw.text((0, 0), "SECURE DOORLOCK", font=font_medium, fill="white")
            draw.line((0, 16, 128, 16), fill="white")
            
            # System Status
            status_text = system_status[:16]  # Limit length
            draw.text((0, 18), f"St: {status_text}", font=font_small, fill="white")
            
            # Confirmation progress
            if confirmation_data['count'] > 0:
                conf_text = f"Conf: {confirmation_data['count']}/{REQUIRED_CONFIRMATIONS}"
                draw.text((0, 28), conf_text, font=font_small, fill="white")
            
            # Last detected person and confidence
            if last_detected_name != "None":
                draw.text((0, 38), f"User: {last_detected_name[:8]}", font=font_small, fill="white")
                if current_confidence > 0:
                    draw.text((0, 48), f"Conf: {(1-current_confidence)*100:.0f}%", font=font_small, fill="white")
            
            # Light warning
            if light_warning:
                draw.text((80, 18), "LOW LIGHT!", font=font_small, fill="white")
            
            # Relay status
            if relay_active:
                remaining_time = relay_delay - (time.time() - relay_trigger_time)
                if remaining_time > 0:
                    draw.rectangle((0, 56, 128, 64), outline="white", fill="white")
                    draw.text((2, 57), f"UNLOCKED ({remaining_time:.1f}s)", font=font_small, fill="black")
                else:
                    draw.rectangle((0, 56, 128, 64), outline="white")
                    draw.text((2, 57), "UNLOCKED", font=font_small, fill="white")
            else:
                draw.rectangle((0, 56, 128, 64), outline="white")
                draw.text((2, 57), "SECURED", font=font_small, fill="white")
                
    except Exception as e:
        print(f"[ERROR] OLED update failed: {e}")

def process_frame(frame):
    global face_locations, face_encodings, face_names, relay_trigger_time, relay_active
    global last_detected_name, detection_time, system_status, current_confidence, light_warning
    
    # Calculate frame brightness
    frame_brightness = calculate_brightness(frame)
    light_warning = frame_brightness < 50
    
    # Resize the frame using cv_scaler to increase performance
    resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
    
    # Convert the image from BGR to RGB colour space
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations, model='large')
    
    face_names = []
    authorized_face_detected = False
    current_confidence = 0.0
    
    if len(face_encodings) > 0:
        system_status = "Analyzing..."
    else:
        system_status = "Scanning..."
        # Reset confirmation if no face detected
        if confirmation_data['count'] > 0:
            confirmation_data['count'] = 0
    
    for i, face_encoding in enumerate(face_encodings):
        # Calculate face area (scale back to original size)
        face_area = calculate_face_area(face_locations[i]) * (cv_scaler ** 2)
        
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=CONFIDENCE_THRESHOLD)
        name = "Unknown"
        
        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        best_distance = face_distances[best_match_index]
        current_confidence = best_distance
        
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            
            # Check detection quality
            is_quality, quality_info = is_good_quality_detection(best_distance, face_area, frame_brightness)
            
            if is_quality:
                # Check if this person is authorized
                if name in authorized_names:
                    # Update confirmation tracking
                    confirmed = update_confirmation(name, best_distance, face_area)
                    
                    if confirmed:
                        authorized_face_detected = True
                        system_status = "ACCESS GRANTED"
                        avg_confidence = np.mean(confirmation_data['confidences'])
                        print(f"[SECURITY] Access granted to {name} after {REQUIRED_CONFIRMATIONS} confirmations")
                        print(f"[SECURITY] Average confidence: {(1-avg_confidence)*100:.1f}%")
                    else:
                        system_status = f"Confirming... {confirmation_data['count']}/{REQUIRED_CONFIRMATIONS}"
                else:
                    system_status = "Recognized - Not Authorized"
            else:
                system_status = "Poor Quality Detection"
                if not quality_info['lighting']:
                    system_status = "Insufficient Lighting"
                elif not quality_info['confidence']:
                    system_status = "Low Confidence"
                elif not quality_info['size']:
                    system_status = "Face Too Small/Distant"
        else:
            system_status = "Unknown Person"
            
        face_names.append(name)
        last_detected_name = name
        detection_time = time.strftime("%H:%M:%S")
    
    # Control the GPIO pin with delay logic
    current_time = time.time()
    
    if authorized_face_detected and not relay_active:
        # Authorized face detected with proper confirmations
        output.on()  # Turn on Pin immediately
        relay_trigger_time = current_time
        relay_active = True
        # Reset confirmation after successful access
        confirmation_data['count'] = 0
        print(f"[INFO] Relay activated for {relay_delay} seconds.")
    
    # Check if we need to turn off the relay after the delay
    if relay_active and (current_time - relay_trigger_time >= relay_delay):
        output.off()  # Turn off Pin after delay
        relay_active = False
        system_status = "Ready"
        print("[INFO] Relay deactivated after delay.")
    
    return frame

def draw_results(frame):
    global current_confidence, light_warning
    
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler
        
        # Calculate face area for display
        face_area = (right - left) * (bottom - top)
        
        # Choose color based on authorization and quality
        if name in authorized_names and face_area >= MIN_FACE_AREA and current_confidence <= CONFIDENCE_THRESHOLD:
            box_color = (0, 255, 0)  # Green for good authorized detection
        elif name in authorized_names:
            box_color = (0, 255, 255)  # Yellow for authorized but poor quality
        else:
            box_color = (0, 0, 255)  # Red for unauthorized or unknown
        
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 3)
        
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left - 3, top - 35), (right + 3, top), box_color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)
        
        # Show confidence and confirmation status
        if name in authorized_names:
            confidence_percent = (1 - current_confidence) * 100
            conf_text = f"Conf: {confidence_percent:.1f}%"
            cv2.putText(frame, conf_text, (left + 6, bottom + 23), font, 0.5, (255, 255, 255), 1)
            
            if confirmation_data['name'] == name:
                conf_status = f"Confirmations: {confirmation_data['count']}/{REQUIRED_CONFIRMATIONS}"
                cv2.putText(frame, conf_status, (left + 6, bottom + 40), font, 0.5, (0, 255, 255), 1)
    
    # Show system warnings
    y_pos = 30
    if light_warning:
        cv2.putText(frame, "WARNING: LOW LIGHT CONDITIONS", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_pos += 25
    
    # Show relay status on screen
    if relay_active:
        remaining_time = relay_delay - (time.time() - relay_trigger_time)
        if remaining_time > 0:
            cv2.putText(frame, f"DOOR UNLOCKED - {remaining_time:.1f}s", (10, y_pos + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "DOOR UNLOCKED", (10, y_pos + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame

def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps

def oled_update_thread():
    """Separate thread for updating OLED display"""
    while True:
        update_oled_display()
        time.sleep(0.1)  # Update display 10 times per second

# Start OLED update thread
if device:
    oled_thread = threading.Thread(target=oled_update_thread, daemon=True)
    oled_thread.start()

# Display startup message on OLED
if device:
    with canvas(device) as draw:
        draw.text((5, 15), "SECURE DOORLOCK", font=font_large, fill="white")
        draw.text((20, 35), "INITIALIZING...", font=font_medium, fill="white")
    time.sleep(2)

print("[INFO] Enhanced Security System Ready!")
print(f"[INFO] Security Settings:")
print(f"       - Required Confirmations: {REQUIRED_CONFIRMATIONS}")
print(f"       - Confirmation Window: {CONFIRMATION_WINDOW}s")
print(f"       - Confidence Threshold: {CONFIDENCE_THRESHOLD}")
print(f"       - Minimum Face Area: {MIN_FACE_AREA} pixels")

while True:
    # Capture a frame from camera
    frame = picam2.capture_array()
    
    # Process the frame with enhanced security
    processed_frame = process_frame(frame)
    
    # Get the text and boxes to be drawn based on the processed frame
    display_frame = draw_results(processed_frame)
    
    # Calculate and update FPS
    current_fps = calculate_fps()
    
    # Attach FPS counter to the text and boxes
    cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (display_frame.shape[1] - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display everything over the video feed.
    cv2.imshow('Video', display_frame)
    
    # Break the loop and stop the script if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# By breaking the loop we run this code here which closes everything
cv2.destroyAllWindows()
picam2.stop()
output.off()  # Make sure to turn off the GPIO pin when exiting

# Clear OLED display on exit
if device:
    with canvas(device) as draw:
        draw.text((35, 20), "SYSTEM", font=font_large, fill="white")
        draw.text((25, 40), "SECURED", font=font_medium, fill="white")
