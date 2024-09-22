import numpy as np
import cv2

def distance(segment_angle, segment_length):
    R = (segment_length/4) * np.tan(segment_angle/4) + (segment_length*segment_length) / (4*segment_length*np.tan(segment_angle/4))

    d = np.cos(segment_angle/2) * R

    return d

# Given two sides with length d1, d2 and an angle alpha between them, 
#  return the length of the third side of the triangle they make 
def cosine_theorem_third_side(d1, d2, alpha):
    return np.sqrt(d1*d1 + d2*d2 - 2*d1*d2*np.cos(alpha))

def plot_triangle(a, b, c):
    # Check if a triangle is valid with the given sides
    if a + b <= c or a + c <= b or b + c <= a:
        print("Invalid triangle sides.")
        return
    
    # Coordinates of the first point
    A = (100, 100)  # Starting point A

    # Calculate the angle at point B using Law of Cosines
    angle_C = np.acos((a**2 + b**2 - c**2) / (2 * a * b))
    angle_C_degrees = np.degrees(angle_C)

    # Point B on the x-axis at distance `a` from A
    B = (A[0] + a, A[1])

    # Calculate the coordinates of point C using trigonometry
    C_x = A[0] + b * np.cos(angle_C)
    C_y = A[1] + b * np.sin(angle_C)
    C = (int(C_x), int(C_y))

    # Create a blank image
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    img.fill(255)  # White background

    # Draw the triangle
    pts = np.array([A, B, C], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 0), thickness=2)

    # Mark the points A, B, C
    cv2.circle(img, A, 5, (0, 0, 255), -1)
    cv2.circle(img, B, 5, (0, 0, 255), -1)
    cv2.circle(img, C, 5, (0, 0, 255), -1)

    # Add text labels
    cv2.putText(img, "A", A, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, "B", B, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, "C", C, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the image
    cv2.imshow("Triangle", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def main():
    tower_h = 95
    segment_angle1 = 10 # degrees
    segment_angle2 = 20 # degrees

    towers_angle = 40 # degrees

    # Calculate the distance from the camera to the tower
    segment_angle1 = np.radians(segment_angle1)
    segment_angle2 = np.radians(segment_angle2)

    d1 = distance(segment_angle1, tower_h)
    d2 = distance(segment_angle2, tower_h)
    d3 = cosine_theorem_third_side(d1, d2, towers_angle)

    d1 = (int)(d1/2)
    d2 = (int)(d2/2)
    d3 = (int)(d3/2)

    print("D1: ", d1)
    print("D2: ", d2)
    print("D3: ", d3)
    plot_triangle(d1, d2, d3)

if __name__ == "__main__":
    main()



