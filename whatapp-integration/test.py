import cv2
import numpy as np
from dataclasses import dataclass


# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:

    # Image
    resize_width: int = 1280

    # Edge detection
    canny_low: int = 40
    canny_high: int = 120

    # Hough
    hough_threshold: int = 60
    min_line_length_ratio: float = 0.08
    max_line_gap: int = 30

    # Morphology
    morphology_kernel: int = 7

    # Region
    min_component_area_ratio: float = 0.002

    # Confidence
    minimum_confidence: float = 0.45


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

class ImagePreprocessor:

    def __init__(self, config):
        self.cfg = config

    def resize(self, image):

        h, w = image.shape[:2]

        if w <= self.cfg.resize_width:
            return image

        scale = self.cfg.resize_width / w

        new_w = self.cfg.resize_width
        new_h = int(h * scale)

        return cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )

    def normalize(self, image):

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        l = clahe.apply(l)

        lab = cv2.merge([l, a, b])

        return cv2.cvtColor(
            lab,
            cv2.COLOR_LAB2BGR
        )

    def process(self, image):

        image = self.resize(image)
        image = self.normalize(image)

        return image


# ============================================================
# EDGE DETECTION
# ============================================================

class EdgeDetector:

    def __init__(self, config):
        self.cfg = config

    def detect(self, image):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        edges = cv2.Canny(
            gray,
            self.cfg.canny_low,
            self.cfg.canny_high
        )

        kernel = np.ones(
            (3, 3),
            np.uint8
        )

        edges = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            kernel
        )

        return edges


# ============================================================
# LINE DETECTION
# ============================================================

@dataclass
class Line:

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def length(self):

        return np.sqrt(
            (self.x2 - self.x1) ** 2 +
            (self.y2 - self.y1) ** 2
        )

    @property
    def angle(self):

        return np.degrees(
            np.arctan2(
                self.y2 - self.y1,
                self.x2 - self.x1
            )
        )


class LineDetector:

    def __init__(self, config):
        self.cfg = config

    def detect(self, edges):

        h, w = edges.shape

        min_length = max(
            30,
            int(w * self.cfg.min_line_length_ratio)
        )

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.cfg.hough_threshold,
            minLineLength=min_length,
            maxLineGap=self.cfg.max_line_gap
        )

        if lines is None:
            return []

        result = []

        # Flatten everything to Nx4
        lines = np.asarray(lines).reshape(-1, 4)

        for x1, y1, x2, y2 in lines:

            result.append(
                Line(
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                )
            )

        return result


# ============================================================
# LINE CLASSIFICATION
# ============================================================

class StructuralAnalyzer:

    def __init__(self):

        pass

    @staticmethod
    def normalize_angle(angle):

        angle = angle % 180

        if angle > 90:
            angle -= 180

        return angle

    def classify(self, lines):

        horizontal = []
        vertical = []
        diagonal = []

        for line in lines:

            angle = self.normalize_angle(
                line.angle
            )

            if abs(angle) < 12:

                horizontal.append(line)

            elif abs(abs(angle) - 90) < 12:

                vertical.append(line)

            else:

                diagonal.append(line)

        return {
            "horizontal": horizontal,
            "vertical": vertical,
            "diagonal": diagonal
        }


# ============================================================
# HORIZONTAL STRUCTURE ANALYSIS
# ============================================================

class FloorBoundaryDetector:

    def __init__(self):

        pass

    def detect(self, horizontal_lines, image_shape):

        h, w = image_shape[:2]

        candidates = []

        for line in horizontal_lines:

            y = int(
                (line.y1 + line.y2) / 2
            )

            # Floor boundary normally exists
            # in lower-middle portion of image.
            normalized_y = y / h

            if 0.30 < normalized_y < 0.90:

                candidates.append(
                    (y, line.length)
                )

        if not candidates:
            return None

        # Prefer longer lines
        candidates.sort(
            key=lambda x: x[1],
            reverse=True
        )

        # Avoid ceiling lines by preferring
        # lower structural lines.
        best = max(
            candidates,
            key=lambda x:
                x[1] * (x[0] / h)
        )

        return best[0]


# ============================================================
# VANISHING POINT
# ============================================================

class VanishingPointEstimator:

    def intersection(self, line1, line2):

        x1, y1 = line1.x1, line1.y1
        x2, y2 = line1.x2, line1.y2

        x3, y3 = line2.x1, line2.y1
        x4, y4 = line2.x2, line2.y2

        denominator = (
            (x1 - x2) * (y3 - y4)
            -
            (y1 - y2) * (x3 - x4)
        )

        if abs(denominator) < 1e-6:
            return None

        px = (
            (x1 * y2 - y1 * x2) * (x3 - x4)
            -
            (x1 - x2) *
            (x3 * y4 - y3 * x4)
        ) / denominator

        py = (
            (x1 * y2 - y1 * x2) * (y3 - y4)
            -
            (y1 - y2) *
            (x3 * y4 - y3 * x4)
        ) / denominator

        return np.array([px, py])

    def estimate(self, diagonal_lines):

        points = []

        for i in range(len(diagonal_lines)):

            for j in range(i + 1, len(diagonal_lines)):

                l1 = diagonal_lines[i]
                l2 = diagonal_lines[j]

                p = self.intersection(l1, l2)

                if p is None:
                    continue

                if np.all(np.isfinite(p)):

                    points.append(p)

        if not points:
            return None

        points = np.array(points)

        return np.median(
            points,
            axis=0
        )


# ============================================================
# COLOR / TEXTURE ANALYSIS
# ============================================================

class AppearanceAnalyzer:

    def convert(self, image):

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV
        )

        lab = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB
        )

        return hsv, lab

    def texture_map(self, image):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        mean = cv2.blur(
            gray.astype(np.float32),
            (15, 15)
        )

        mean_sq = cv2.blur(
            gray.astype(np.float32) ** 2,
            (15, 15)
        )

        variance = (
            mean_sq -
            mean ** 2
        )

        variance = cv2.normalize(
            variance,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        return variance.astype(np.uint8)


# ============================================================
# INITIAL ROOM MASK
# ============================================================

class RoomMaskBuilder:

    def create_wall_mask(
        self,
        image_shape,
        floor_y
    ):

        h, w = image_shape[:2]

        mask = np.zeros(
            (h, w),
            dtype=np.uint8
        )

        if floor_y is None:

            floor_y = int(
                h * 0.65
            )

        polygon = np.array([
            [0, 0],
            [w, 0],
            [w, floor_y],
            [0, floor_y]
        ])

        cv2.fillPoly(
            mask,
            [polygon],
            255
        )

        return mask

    def create_floor_mask(
        self,
        image_shape,
        floor_y
    ):

        h, w = image_shape[:2]

        mask = np.zeros(
            (h, w),
            dtype=np.uint8
        )

        if floor_y is None:

            floor_y = int(
                h * 0.65
            )

        polygon = np.array([
            [0, floor_y],
            [w, floor_y],
            [w, h],
            [0, h]
        ])

        cv2.fillPoly(
            mask,
            [polygon],
            255
        )

        return mask


# ============================================================
# MASK REFINEMENT
# ============================================================

class MaskRefiner:

    def __init__(self, config):

        self.cfg = config

    def refine(self, mask):

        k = self.cfg.morphology_kernel

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (k, k)
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        return mask


# ============================================================
# CONFIDENCE
# ============================================================

class ConfidenceEstimator:

    def calculate(
        self,
        floor_y,
        image_shape,
        lines
    ):

        h, w = image_shape[:2]

        score = 0.0

        # Structural line evidence
        if len(lines) > 0:
            score += 0.35

        # Reasonable floor boundary
        if floor_y is not None:

            normalized = floor_y / h

            if 0.45 < normalized < 0.85:

                score += 0.40

        # Basic geometric consistency
        if floor_y is not None:

            if floor_y < h:

                score += 0.25

        return min(score, 1.0)


# ============================================================
# VISUALIZATION
# ============================================================

class Visualizer:

    def overlay(
        self,
        image,
        wall_mask,
        floor_mask
    ):

        output = image.copy()

        wall_color = np.zeros_like(
            image
        )

        floor_color = np.zeros_like(
            image
        )

        wall_color[:, :] = (
            255,
            0,
            0
        )

        floor_color[:, :] = (
            0,
            255,
            0
        )

        wall_region = (
            wall_mask > 0
        )

        floor_region = (
            floor_mask > 0
        )

        output[wall_region] = cv2.addWeighted(
            image[wall_region],
            0.55,
            wall_color[wall_region],
            0.45,
            0
        )

        output[floor_region] = cv2.addWeighted(
            image[floor_region],
            0.55,
            floor_color[floor_region],
            0.45,
            0
        )

        return output


# ============================================================
# MAIN PIPELINE
# ============================================================

class InteriorSegmenter:

    def __init__(self):

        self.cfg = Config()

        self.preprocessor = \
            ImagePreprocessor(self.cfg)

        self.edge_detector = \
            EdgeDetector(self.cfg)

        self.line_detector = \
            LineDetector(self.cfg)

        self.structural_analyzer = \
            StructuralAnalyzer()

        self.floor_detector = \
            FloorBoundaryDetector()

        self.vanishing_estimator = \
            VanishingPointEstimator()

        self.appearance = \
            AppearanceAnalyzer()

        self.mask_builder = \
            RoomMaskBuilder()

        self.refiner = \
            MaskRefiner(self.cfg)

        self.confidence = \
            ConfidenceEstimator()

        self.visualizer = \
            Visualizer()

    def process(self, image):

        # ----------------------------------------
        # 1. PREPROCESS
        # ----------------------------------------

        image = self.preprocessor.process(
            image
        )

        # ----------------------------------------
        # 2. EDGES
        # ----------------------------------------

        edges = self.edge_detector.detect(
            image
        )

        # ----------------------------------------
        # 3. LINES
        # ----------------------------------------

        lines = self.line_detector.detect(
            edges
        )

        # ----------------------------------------
        # 4. STRUCTURAL ANALYSIS
        # ----------------------------------------

        groups = self.structural_analyzer.classify(
            lines
        )

        # ----------------------------------------
        # 5. FLOOR BOUNDARY
        # ----------------------------------------

        floor_y = self.floor_detector.detect(
            groups["horizontal"],
            image.shape
        )

        # ----------------------------------------
        # 6. VANISHING POINT
        # ----------------------------------------

        vanishing_point = \
            self.vanishing_estimator.estimate(
                groups["diagonal"]
            )

        # ----------------------------------------
        # 7. APPEARANCE
        # ----------------------------------------

        hsv, lab = self.appearance.convert(
            image
        )

        texture = self.appearance.texture_map(
            image
        )

        # ----------------------------------------
        # 8. INITIAL MASKS
        # ----------------------------------------

        wall_mask = \
            self.mask_builder.create_wall_mask(
                image.shape,
                floor_y
            )

        floor_mask = \
            self.mask_builder.create_floor_mask(
                image.shape,
                floor_y
            )

        # ----------------------------------------
        # 9. REFINEMENT
        # ----------------------------------------

        wall_mask = self.refiner.refine(
            wall_mask
        )

        floor_mask = self.refiner.refine(
            floor_mask
        )

        # ----------------------------------------
        # 10. CONFIDENCE
        # ----------------------------------------

        confidence = \
            self.confidence.calculate(
                floor_y,
                image.shape,
                lines
            )

        # ----------------------------------------
        # 11. VISUALIZATION
        # ----------------------------------------

        overlay = self.visualizer.overlay(
            image,
            wall_mask,
            floor_mask
        )

        return {
            "image": image,
            "edges": edges,
            "lines": lines,
            "vanishing_point": vanishing_point,
            "floor_y": floor_y,
            "wall_mask": wall_mask,
            "floor_mask": floor_mask,
            "texture": texture,
            "confidence": confidence,
            "overlay": overlay
        }


# ============================================================
# ENTRY POINT
# ============================================================

def main():

    image = cv2.imread(
        "ambigous case.jpg"
    )

    if image is None:

        raise FileNotFoundError(
            "data/room.jpg not found"
        )

    segmenter = InteriorSegmenter()

    result = segmenter.process(
        image
    )

    print(
        f"Confidence: "
        f"{result['confidence']:.2f}"
    )

    cv2.imwrite(
        "edges.jpg",
        result["edges"]
    )

    cv2.imwrite(
        "wall_mask.png",
        result["wall_mask"]
    )

    cv2.imwrite(
        "floor_mask.png",
        result["floor_mask"]
    )

    cv2.imwrite(
        "segmentation_result.jpg",
        result["overlay"]
    )

    cv2.imshow(
        "Segmentation",
        result["overlay"]
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()