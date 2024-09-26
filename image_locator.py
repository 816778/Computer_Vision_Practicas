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

def plot_triangle(a, b, c, scale=1):
    # Check if a triangle is valid with the given sides
    if a + b <= c or a + c <= b or b + c <= a:
        print("Invalid triangle sides.")
        return
    
    a = int(a * scale)
    b = int(b * scale)
    c = int(c * scale)


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


def decompose_fov(diagonal_fov, aspect_ratio_w, aspect_ratio_h):

    Ha = np.radians(aspect_ratio_w)
    Va = np.radians(aspect_ratio_h)
    Df = np.radians(diagonal_fov)

    Da = np.sqrt(Ha*Ha + Va*Va)

    Hf = np.arctan(np.tan(Df/2) * (Ha/Da)) * 2
    Vf = np.arctan(np.tan(Df/2) * (Va/Da)) * 2

    return np.degrees(Hf), np.degrees(Vf)


def load_image(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Resize the image
    img = cv2.resize(img, (1280, 720))

    # Display the image
    cv2.imshow("Image", img)

    return img

def capture_points(img):

    points = []

    # Mouse event callback function
    def mouse_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("Clicked at: ", x, y)

            points.append((x, y))

            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Image", img)

    # Set the mouse callback function
    cv2.setMouseCallback("Image", mouse_event)

    # Wait for the user to click on the image
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return np.array(points)

def camera_horizontal_alpha(relative_width, dmin):
    
    return 2 * np.acos(relative_width/(2*dmin))



def main():
    tower_h = 95
    diagonal_fov = 83.259

    # Load the image
    img = load_image("img/103.jpg")
    img_height, img_width, _ = img.shape

    # Calculate the horizontal and vertical FOV
    Hf, Vf = decompose_fov(diagonal_fov, 16, 9)


    # Capture the points
    points = capture_points(img)

    vt1 = points[0] - points[1]
    vt2 = points[2] - points[3]
    vt1t2 = points[0] - points[2]

    dt1 = np.linalg.norm(vt1)
    dt2 = np.linalg.norm(vt2)
    dt1t2 = np.abs(vt1t2[0])

    # Raster space to image space
    dt1 = dt1 / img_height
    dt2 = dt2 / img_height
    dt1t2 = dt1t2 / img_width

    print("dt1: ", dt1)
    print("dt2: ", dt2)
    print("dt1t2: ", dt1t2)

    # Calculate the angle of the segment
    segment_angle1 = Vf * dt1
    segment_angle2 = Vf * dt2    
    towers_angle = Hf * dt1t2

    
    # Calculate the distance from the camera to the tower
    segment_angle1 = np.radians(segment_angle1)
    segment_angle2 = np.radians(segment_angle2)
    towers_angle = np.radians(towers_angle) # Camera horizontal angle returns in radians

    d1 = distance(segment_angle1, tower_h)
    d2 = distance(segment_angle2, tower_h)

    #if d1 < d2:
    #    dmin = np.minimum(d1, d2)
    #    k_ = (dt1t2 / (np.abs(vt1[1]) / img_height)) * dmin
    #else:
    #    dmin = np.minimum(d1, d2)
    #    k_ = (dt1t2 / (np.abs(vt2[1]) / img_width)) * dmin
#
    #towers_angle = camera_horizontal_alpha(k_, dmin)
    #print("Towers angle:", towers_angle)
    
    d3 = cosine_theorem_third_side(d1, d2, towers_angle)

    print("D1: ", d1)
    print("D2: ", d2)
    print("D3: ", d3)
    plot_triangle(d1, d2, d3, 1)

if __name__ == "__main__":
    main()



