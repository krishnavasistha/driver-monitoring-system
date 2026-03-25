class HeadPoseEstimator:
    def __init__(self, frame_width):
        self.frame_width = frame_width

    def estimate(self, mesh_points):
        if mesh_points is None or len(mesh_points) <= 1:
            return "Forward"

        nose_tip = mesh_points[1]
        if nose_tip[0] < self.frame_width // 3:
            return "Left"
        elif nose_tip[0] > self.frame_width * 2 // 3:
            return "Right"
        return "Forward"
