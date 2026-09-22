import cv2
import numpy as np

from dataclasses import dataclass
from skimage.segmentation import slic


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:

    # Processing
    max_width: int = 1280

    # SLIC
    n_segments: int = 900
    compactness: float = 12.0

    # Canny
    canny_low: int = 40
    canny_high: int = 130

    # Hough
    hough_threshold: int = 45
    min_line_length: int = 45
    max_line_gap: int = 25

    # Boundary search
    boundary_min_ratio: float = 0.35
    boundary_max_ratio: float = 0.90

    # Morphology
    morphology_size: int = 5

    # Minimum component
    min_component_ratio: float = 0.0015

    # Visualization
    overlay_alpha: float = 0.45


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

class ImagePreprocessor:

    def __init__(self, cfg):
        self.cfg = cfg

    def resize(self, image):

        h, w = image.shape[:2]

        if w <= self.cfg.max_width:
            return image

        scale = self.cfg.max_width / w

        new_size = (
            self.cfg.max_width,
            int(h * scale)
        )

        return cv2.resize(
            image,
            new_size,
            interpolation=cv2.INTER_AREA
        )

    def illumination_normalization(self, image):

        lab = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB
        )

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        l = clahe.apply(l)

        return cv2.cvtColor(
            cv2.merge((l, a, b)),
            cv2.COLOR_LAB2BGR
        )

    def process(self, image):

        image = self.resize(image)

        image = self.illumination_normalization(
            image
        )

        return image


# ============================================================
# FEATURE EXTRACTION
# ============================================================

class FeatureExtractor:

    def __init__(self):
        pass

    def lab(self, image):

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB
        )

    def hsv(self, image):

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV
        )

    def grayscale(self, image):

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    def edges(self, image, cfg):

        gray = self.grayscale(image)

        gray = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        return cv2.Canny(
            gray,
            cfg.canny_low,
            cfg.canny_high
        )

    def gradient(self, image):

        gray = self.grayscale(image)

        gx = cv2.Sobel(
            gray,
            cv2.CV_32F,
            1,
            0,
            ksize=3
        )

        gy = cv2.Sobel(
            gray,
            cv2.CV_32F,
            0,
            1,
            ksize=3
        )

        magnitude = cv2.magnitude(
            gx,
            gy
        )

        magnitude = cv2.normalize(
            magnitude,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        return magnitude.astype(
            np.uint8
        )

    def texture(self, image):

        gray = self.grayscale(image)

        gray = gray.astype(
            np.float32
        )

        mean = cv2.blur(
            gray,
            (15, 15)
        )

        mean_square = cv2.blur(
            gray * gray,
            (15, 15)
        )

        variance = (
            mean_square -
            mean * mean
        )

        variance = np.maximum(
            variance,
            0
        )

        variance = cv2.normalize(
            variance,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        return variance.astype(
            np.uint8
        )


# ============================================================
# EDGE / LINE ANALYSIS
# ============================================================

class StructuralAnalyzer:

    def __init__(self, cfg):
        self.cfg = cfg

    def detect_lines(self, edges):

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.cfg.hough_threshold,
            minLineLength=self.cfg.min_line_length,
            maxLineGap=self.cfg.max_line_gap
        )

        if lines is None:
            return []

        return lines.reshape(
            -1,
            4
        )

    def angle(self, line):

        x1, y1, x2, y2 = line

        return np.degrees(
            np.arctan2(
                y2 - y1,
                x2 - x1
            )
        )

    def length(self, line):

        x1, y1, x2, y2 = line

        return np.hypot(
            x2 - x1,
            y2 - y1
        )

    def horizontal_lines(
        self,
        lines
    ):

        result = []

        for line in lines:

            angle = self.angle(
                line
            )

            normalized = abs(
                ((angle + 90) % 180) - 90
            )

            if normalized <= 12:

                result.append(line)

        return result

    def diagonal_lines(
        self,
        lines
    ):

        result = []

        for line in lines:

            angle = self.angle(
                line
            )

            normalized = abs(
                ((angle + 90) % 180) - 90
            )

            if (
                15 <= normalized <= 75
            ):

                result.append(line)

        return result


# ============================================================
# SUPERPIXEL SEGMENTATION
# ============================================================

class SuperpixelAnalyzer:

    def __init__(self, cfg):
        self.cfg = cfg

    def create(self, image):

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        labels = slic(
            rgb,
            n_segments=self.cfg.n_segments,
            compactness=self.cfg.compactness,
            start_label=0
        )

        return labels

    def region_statistics(
        self,
        image,
        labels
    ):

        lab = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB
        )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        count = labels.max() + 1

        stats = []

        for label_id in range(count):

            mask = labels == label_id

            if not np.any(mask):
                continue

            mean_lab = np.mean(
                lab[mask],
                axis=0
            )

            mean_gray = np.mean(
                gray[mask]
            )

            texture = np.var(
                gray[mask]
            )

            ys, xs = np.where(mask)

            stats.append({
                "id": label_id,
                "mean_lab": mean_lab,
                "mean_gray": mean_gray,
                "texture": texture,
                "center": (
                    float(np.mean(xs)),
                    float(np.mean(ys))
                ),
                "area": int(np.sum(mask))
            })

        return stats


# ============================================================
# VANISHING POINT ESTIMATION
# ============================================================

class VanishingPointEstimator:

    @staticmethod
    def intersection(l1, l2):

        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2

        denominator = (
            (x1 - x2) * (y3 - y4)
            -
            (y1 - y2) * (x3 - x4)
        )

        if abs(denominator) < 1e-8:
            return None

        px = (
            (x1 * y2 - y1 * x2)
            * (x3 - x4)
            -
            (x1 - x2)
            * (x3 * y4 - y3 * x4)
        ) / denominator

        py = (
            (x1 * y2 - y1 * x2)
            * (y3 - y4)
            -
            (y1 - y2)
            * (x3 * y4 - y3 * x4)
        ) / denominator

        return np.array(
            [px, py],
            dtype=np.float32
        )

    def estimate(self, lines):

        if len(lines) < 2:
            return None

        intersections = []

        for i in range(len(lines)):

            for j in range(
                i + 1,
                len(lines)
            ):

                p = self.intersection(
                    lines[i],
                    lines[j]
                )

                if p is None:
                    continue

                if np.all(
                    np.isfinite(p)
                ):

                    intersections.append(
                        p
                    )

        if len(intersections) < 3:
            return None

        intersections = np.asarray(
            intersections
        )

        # Remove extreme outliers
        median = np.median(
            intersections,
            axis=0
        )

        distance = np.linalg.norm(
            intersections - median,
            axis=1
        )

        threshold = np.percentile(
            distance,
            60
        )

        valid = intersections[
            distance <= threshold
        ]

        if len(valid) == 0:
            return None

        return np.mean(
            valid,
            axis=0
        )


# ============================================================
# FLOOR BOUNDARY CANDIDATES
# ============================================================

class BoundaryAnalyzer:

    def __init__(self, cfg):
        self.cfg = cfg

    def generate_candidates(
        self,
        image,
        edges,
        gradient,
        horizontal_lines
    ):

        h, w = image.shape[:2]

        y_min = int(
            h * self.cfg.boundary_min_ratio
        )

        y_max = int(
            h * self.cfg.boundary_max_ratio
        )

        candidates = []

        # ----------------------------------------------------
        # 1. Hough horizontal evidence
        # ----------------------------------------------------

        hough_score = np.zeros(
            h,
            dtype=np.float32
        )

        for line in horizontal_lines:

            x1, y1, x2, y2 = line

            y = int(
                (y1 + y2) / 2
            )

            if y < 0 or y >= h:
                continue

            length = np.hypot(
                x2 - x1,
                y2 - y1
            )

            hough_score[y] += length

        if hough_score.max() > 0:

            hough_score /= (
                hough_score.max()
            )

        # ----------------------------------------------------
        # 2. Edge evidence
        # ----------------------------------------------------

        edge_score = np.mean(
            edges > 0,
            axis=1
        ).astype(
            np.float32
        )

        if edge_score.max() > 0:

            edge_score /= (
                edge_score.max()
            )

        # ----------------------------------------------------
        # 3. Gradient evidence
        # ----------------------------------------------------

        gradient_score = np.mean(
            gradient,
            axis=1
        )

        if gradient_score.max() > 0:

            gradient_score /= (
                gradient_score.max()
            )

        # ----------------------------------------------------
        # 4. Combined score
        # ----------------------------------------------------

        for y in range(
            y_min,
            y_max
        ):

            score = (
                0.50 * hough_score[y]
                +
                0.30 * edge_score[y]
                +
                0.20 * gradient_score[y]
            )

            candidates.append(
                (
                    y,
                    float(score)
                )
            )

        candidates.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return candidates


# ============================================================
# DYNAMIC BOUNDARY
# ============================================================

class DynamicBoundary:

    def __init__(self, cfg):
        self.cfg = cfg

    def create(
        self,
        image,
        edges,
        candidate_y
    ):

        h, w = image.shape[:2]

        # ----------------------------------------------------
        # Search local edge maxima around candidate
        # ----------------------------------------------------

        points = []

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        gradient_x = cv2.Sobel(
            gray,
            cv2.CV_32F,
            0,
            1,
            ksize=3
        )

        gradient_x = np.abs(
            gradient_x
        )

        # Divide image into vertical strips
        # and find strongest boundary.
        number_of_points = 25

        xs = np.linspace(
            0,
            w - 1,
            number_of_points
        ).astype(int)

        search_radius = max(
            12,
            int(h * 0.08)
        )

        for x in xs:

            x0 = max(
                0,
                x - 3
            )

            x1 = min(
                w,
                x + 4
            )

            local = gradient_x[
                :,
                x0:x1
            ]

            score = np.mean(
                local,
                axis=1
            )

            low = max(
                0,
                candidate_y - search_radius
            )

            high = min(
                h,
                candidate_y + search_radius
            )

            if high <= low:
                continue

            local_score = score[
                low:high
            ]

            if len(local_score) == 0:
                continue

            best_offset = np.argmax(
                local_score
            )

            y = low + best_offset

            points.append(
                (x, y)
            )

        if len(points) < 5:

            return np.array([
                [0, candidate_y],
                [w - 1, candidate_y]
            ])

        points = np.asarray(
            points,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Robust polynomial fit
        # ----------------------------------------------------

        x = points[:, 0]
        y = points[:, 1]

        degree = 2

        coeff = np.polyfit(
            x,
            y,
            degree
        )

        fitted_x = np.linspace(
            0,
            w - 1,
            w
        )

        fitted_y = np.polyval(
            coeff,
            fitted_x
        )

        fitted_y = np.clip(
            fitted_y,
            0,
            h - 1
        )

        return np.column_stack(
            [fitted_x, fitted_y]
        )


# ============================================================
# MASK BUILDER
# ============================================================

class MaskBuilder:

    def wall_mask(
        self,
        image_shape,
        boundary
    ):

        h, w = image_shape[:2]

        mask = np.zeros(
            (h, w),
            dtype=np.uint8
        )

        polygon = np.vstack([
            [0, 0],
            [w - 1, 0],
            boundary[::-1],
            [0, 0]
        ])

        polygon = polygon.astype(
            np.int32
        )

        cv2.fillPoly(
            mask,
            [polygon],
            255
        )

        return mask

    def floor_mask(
        self,
        image_shape,
        boundary
    ):

        h, w = image_shape[:2]

        mask = np.zeros(
            (h, w),
            dtype=np.uint8
        )

        polygon = np.vstack([
            boundary,
            [w - 1, h - 1],
            [0, h - 1]
        ])

        polygon = polygon.astype(
            np.int32
        )

        cv2.fillPoly(
            mask,
            [polygon],
            255
        )

        return mask


# ============================================================
# REGION REFINEMENT
# ============================================================

class RegionRefiner:

    def __init__(self, cfg):
        self.cfg = cfg

    def morphology(self, mask):

        k = self.cfg.morphology_size

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

    def remove_small_components(
        self,
        mask
    ):

        h, w = mask.shape

        minimum_area = int(
            h * w *
            self.cfg.min_component_ratio
        )

        num_labels, labels, stats, _ = \
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8
            )

        result = np.zeros_like(
            mask
        )

        for i in range(
            1,
            num_labels
        ):

            area = stats[
                i,
                cv2.CC_STAT_AREA
            ]

            if area >= minimum_area:

                result[
                    labels == i
                ] = 255

        return result

    def refine(self, mask):

        mask = self.morphology(
            mask
        )

        mask = self.remove_small_components(
            mask
        )

        return mask


# ============================================================
# CONFIDENCE
# ============================================================

class ConfidenceEstimator:

    def calculate(
        self,
        candidate_score,
        boundary,
        image_shape,
        lines
    ):

        h, w = image_shape[:2]

        score = 0.0

        # Boundary confidence
        score += min(
            candidate_score,
            1.0
        ) * 0.45

        # Structural evidence
        line_score = min(
            len(lines) / 20.0,
            1.0
        )

        score += (
            line_score * 0.25
        )

        # Boundary smoothness
        if len(boundary) > 3:

            second_derivative = \
                np.diff(
                    boundary[:, 1],
                    n=2
                )

            smoothness = 1.0 / (
                1.0 +
                np.mean(
                    np.abs(
                        second_derivative
                    )
                )
            )

            score += (
                smoothness * 0.30
            )

        return float(
            np.clip(
                score,
                0,
                1
            )
        )


# ============================================================
# VISUALIZATION
# ============================================================

class Visualizer:

    def draw_boundary(
        self,
        image,
        boundary
    ):

        result = image.copy()

        points = boundary.astype(
            np.int32
        ).reshape(
            -1,
            1,
            2
        )

        cv2.polylines(
            result,
            [points],
            False,
            (0, 0, 255),
            3
        )

        return result

    def overlay(
        self,
        image,
        wall_mask,
        floor_mask,
        boundary
    ):

        result = image.copy()

        wall_layer = np.zeros_like(
            image
        )

        floor_layer = np.zeros_like(
            image
        )

        # BGR
        wall_layer[:] = (
            255,
            80,
            50
        )

        floor_layer[:] = (
            50,
            220,
            80
        )

        wall_region = (
            wall_mask > 0
        )

        floor_region = (
            floor_mask > 0
        )

        result[wall_region] = cv2.addWeighted(
            image[wall_region],
            1.0 - 0.40,
            wall_layer[wall_region],
            0.40,
            0
        )

        result[floor_region] = cv2.addWeighted(
            image[floor_region],
            1.0 - 0.40,
            floor_layer[floor_region],
            0.40,
            0
        )

        result = self.draw_boundary(
            result,
            boundary
        )

        return result


# ============================================================
# MAIN SEGMENTER
# ============================================================

class InteriorSegmenter:

    def __init__(self):

        self.cfg = Config()

        self.preprocessor = \
            ImagePreprocessor(
                self.cfg
            )

        self.features = \
            FeatureExtractor()

        self.structure = \
            StructuralAnalyzer(
                self.cfg
            )

        self.superpixels = \
            SuperpixelAnalyzer(
                self.cfg
            )

        self.vanishing = \
            VanishingPointEstimator()

        self.boundary_analyzer = \
            BoundaryAnalyzer(
                self.cfg
            )

        self.dynamic_boundary = \
            DynamicBoundary(
                self.cfg
            )

        self.mask_builder = \
            MaskBuilder()

        self.refiner = \
            RegionRefiner(
                self.cfg
            )

        self.confidence = \
            ConfidenceEstimator()

        self.visualizer = \
            Visualizer()

    def process(
        self,
        image
    ):

        # ====================================================
        # STEP 1
        # PREPROCESS
        # ====================================================

        image = self.preprocessor.process(
            image
        )

        # ====================================================
        # STEP 2
        # FEATURES
        # ====================================================

        edges = self.features.edges(
            image,
            self.cfg
        )

        gradient = self.features.gradient(
            image
        )

        lab = self.features.lab(
            image
        )

        hsv = self.features.hsv(
            image
        )

        texture = self.features.texture(
            image
        )

        # ====================================================
        # STEP 3
        # STRUCTURAL LINES
        # ====================================================

        lines = self.structure.detect_lines(
            edges
        )

        horizontal = \
            self.structure.horizontal_lines(
                lines
            )

        diagonal = \
            self.structure.diagonal_lines(
                lines
            )

        # ====================================================
        # STEP 4
        # VANISHING POINT
        # ====================================================

        vanishing_point = \
            self.vanishing.estimate(
                diagonal
            )

        # ====================================================
        # STEP 5
        # SUPERPIXELS
        # ====================================================

        superpixel_labels = \
            self.superpixels.create(
                image
            )

        region_stats = \
            self.superpixels.region_statistics(
                image,
                superpixel_labels
            )

        # ====================================================
        # STEP 6
        # CANDIDATE BOUNDARIES
        # ====================================================

        candidates = \
            self.boundary_analyzer.generate_candidates(
                image,
                edges,
                gradient,
                horizontal
            )

        if len(candidates) == 0:

            h, w = image.shape[:2]

            candidate_y = int(
                h * 0.65
            )

            candidate_score = 0.0

        else:

            candidate_y = \
                candidates[0][0]

            candidate_score = \
                candidates[0][1]

        # ====================================================
        # STEP 7
        # DYNAMIC BOUNDARY
        # ====================================================

        boundary = \
            self.dynamic_boundary.create(
                image,
                edges,
                candidate_y
            )

        # ====================================================
        # STEP 8
        # INITIAL MASKS
        # ====================================================

        wall_mask = \
            self.mask_builder.wall_mask(
                image.shape,
                boundary
            )

        floor_mask = \
            self.mask_builder.floor_mask(
                image.shape,
                boundary
            )

        # ====================================================
        # STEP 9
        # REFINEMENT
        # ====================================================

        wall_mask = \
            self.refiner.refine(
                wall_mask
            )

        floor_mask = \
            self.refiner.refine(
                floor_mask
            )

        # ====================================================
        # STEP 10
        # CONFIDENCE
        # ====================================================

        confidence = \
            self.confidence.calculate(
                candidate_score,
                boundary,
                image.shape,
                lines
            )

        # ====================================================
        # STEP 11
        # VISUALIZATION
        # ====================================================

        overlay = \
            self.visualizer.overlay(
                image,
                wall_mask,
                floor_mask,
                boundary
            )

        return {

            "image": image,

            "edges": edges,

            "gradient": gradient,

            "texture": texture,

            "lab": lab,

            "hsv": hsv,

            "lines": lines,

            "horizontal_lines": horizontal,

            "diagonal_lines": diagonal,

            "vanishing_point":
                vanishing_point,

            "superpixel_labels":
                superpixel_labels,

            "region_stats":
                region_stats,

            "boundary":
                boundary,

            "wall_mask":
                wall_mask,

            "floor_mask":
                floor_mask,

            "confidence":
                confidence,

            "overlay":
                overlay
        }


# ============================================================
# SAVE DEBUG INFORMATION
# ============================================================

def save_debug_outputs(
    result
):

    cv2.imwrite(
        "output_edges.png",
        result["edges"]
    )

    cv2.imwrite(
        "output_gradient.png",
        result["gradient"]
    )

    cv2.imwrite(
        "output_texture.png",
        result["texture"]
    )

    cv2.imwrite(
        "output_wall_mask.png",
        result["wall_mask"]
    )

    cv2.imwrite(
        "output_floor_mask.png",
        result["floor_mask"]
    )

    cv2.imwrite(
        "output_segmentation.png",
        result["overlay"]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    image_path = "image_from_different_angel.jpg"

    image = cv2.imread(
        image_path
    )

    if image is None:

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    print(
        f"Input image: "
        f"{image.shape[1]} x "
        f"{image.shape[0]}"
    )

    segmenter = \
        InteriorSegmenter()

    result = segmenter.process(
        image
    )

    print(
        f"Detected lines: "
        f"{len(result['lines'])}"
    )

    print(
        f"Horizontal lines: "
        f"{len(result['horizontal_lines'])}"
    )

    print(
        f"Diagonal lines: "
        f"{len(result['diagonal_lines'])}"
    )

    print(
        f"Confidence: "
        f"{result['confidence']:.3f}"
    )

    if result["vanishing_point"] is not None:

        x, y = result[
            "vanishing_point"
        ]

        print(
            f"Vanishing point: "
            f"({x:.1f}, {y:.1f})"
        )

    save_debug_outputs(
        result
    )

    cv2.imshow(
        "Interior Segmentation",
        result["overlay"]
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()