import cv2

def draw_person_box(frame, bbox, pid, person_data):
    """Draw bounding box and ID/name for a detected person"""
    l, t, r, b = bbox
    color = person_data['color']
    name = person_data.get('name')
    
    display_text = f"ID:{pid}" if name is None else name
    
    cv2.rectangle(frame, (l, t), (r, b), color, 2)
    
    (text_width, text_height), _ = cv2.getTextSize(
        display_text, 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.6, 
        2
    )
    
    cv2.rectangle(
        frame, 
        (l, t - text_height - 10), 
        (l + text_width, t), 
        color, 
        -1
    )
    
    cv2.putText(
        frame, 
        display_text, 
        (l, t - 5), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.6, 
        (255, 255, 255), 
        2
    )
    
    return frame