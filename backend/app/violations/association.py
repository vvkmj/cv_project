from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class RiderAssociator:
    """
    Associates detected people with detected motorcycles based on geometric,
    spatial overlap, and posture relationships.
    """

    def __init__(
        self,
        min_overlap_ratio: float = 0.20,
        max_horizontal_leeway: float = 0.25,
        max_vertical_gap: float = 0.15
    ):
        self.min_overlap_ratio = min_overlap_ratio
        self.max_horizontal_leeway = max_horizontal_leeway
        self.max_vertical_gap = max_vertical_gap

    def is_person_riding_motorcycle(
        self, person_box: List[float], moto_box: List[float]
    ) -> Tuple[bool, float]:
        """
        Determines whether a person is a rider of the motorcycle.
        Returns: (is_rider: bool, score: float)
        Boxes are [x1, y1, x2, y2].
        """
        px1, py1, px2, py2 = person_box
        mx1, my1, mx2, my2 = moto_box

        pw = max(1.0, px2 - px1)
        ph = max(1.0, py2 - py1)
        p_area = pw * ph

        mw = max(1.0, mx2 - mx1)
        mh = max(1.0, my2 - my1)

        # Person center coordinates
        pcx = (px1 + px2) / 2.0

        # Motorcycle expanded horizontal boundaries (riders sit directly above/on motorcycle)
        h_leeway = mw * self.max_horizontal_leeway
        if pcx < (mx1 - h_leeway) or pcx > (mx2 + h_leeway):
            return False, 0.0

        # Vertical validation:
        # A rider's lower body/feet sits inside or near the motorcycle's top half.
        # Person cannot be completely below the motorcycle (that would be a pedestrian or artifact)
        # Person's head (py1) should typically be near or above motorcycle top (my1).
        if py2 < (my1 - mh * self.max_vertical_gap):
            # Person is floating completely above motorcycle
            return False, 0.0
        if py1 + ph * 0.30 > my1 + mh * 0.32 or py2 > my2 + mh * 0.15:
            # Person is almost entirely below motorcycle's top section
            return False, 0.0

        # Calculate intersection area between person and motorcycle
        ix1 = max(px1, mx1)
        iy1 = max(py1, my1)
        ix2 = min(px2, mx2)
        iy2 = min(py2, my2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter_area = iw * ih

        # Overlap ratio relative to person area
        overlap_ratio = inter_area / p_area

        # Alignment score based on horizontal centering and overlap
        center_dist = abs(pcx - (mx1 + mx2) / 2.0) / (mw / 2.0 + 1e-5)
        alignment_score = max(0.0, 1.0 - min(1.0, center_dist * 0.5))
        
        # Combined association score
        association_score = (overlap_ratio * 0.6) + (alignment_score * 0.4)

        # Person qualifies as a rider if there is reasonable overlap or person lower body is seated on motorcycle
        is_rider = (overlap_ratio >= self.min_overlap_ratio) or (
            py2 >= my1 and py1 <= my1 and (mx1 - h_leeway <= pcx <= mx2 + h_leeway)
        )

        return bool(is_rider), float(association_score)

    def extract_head_region(self, person_box: List[float], image_shape: Tuple[int, int]) -> List[int]:
        """
        Extracts the head/helmet bounding box region from a person's bounding box.
        Top 28% of the person's bounding box.
        Returns [hx1, hy1, hx2, hy2] clipped to image dimensions.
        """
        px1, py1, px2, py2 = person_box
        pw = px2 - px1
        ph = py2 - py1

        # Head region is upper 28% of person height, with 5% horizontal inset to focus on head
        hx1 = int(max(0, px1 + pw * 0.05))
        hy1 = int(max(0, py1))
        hx2 = int(min(image_shape[1], px2 - pw * 0.05))
        hy2 = int(min(image_shape[0], py1 + ph * 0.30))

        if hx2 <= hx1:
            hx2 = int(min(image_shape[1], hx1 + 10))
        if hy2 <= hy1:
            hy2 = int(min(image_shape[0], hy1 + 10))

        return [hx1, hy1, hx2, hy2]

    def associate_riders(
        self,
        detections: List[Dict[str, Any]],
        image_shape: Tuple[int, int]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Takes all detections in a frame.
        Identifies motorcycles and persons, associates persons to motorcycles,
        and annotates motorcycles with their rider IDs and head regions.
        """
        motorcycles = [d for d in detections if d["object_class"] == "motorcycle"]
        persons = [d for d in detections if d["object_class"] == "person"]

        keys = ("x1", "y1", "x2", "y2")
        bike_boxes = []
        for moto in motorcycles:
            moto["associated_riders"] = []
            bike_boxes.append([moto["bounding_box"][key] for key in keys])

        for person in persons:
            box = person["bounding_box"]
            person_box = [box[key] for key in keys]
            head_box = dict(zip(keys, self.extract_head_region(person_box, image_shape)))
            person["head_box"] = head_box
            candidates = []
            for index, bike_box in enumerate(bike_boxes):
                is_rider, score = self.is_person_riding_motorcycle(person_box, bike_box)
                if is_rider:
                    candidates.append((score, index))
            candidates.sort(reverse=True)
            if not candidates or (len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.07):
                continue
            score, index = candidates[0]
            motorcycles[index]["associated_riders"].append({
                "person_tracking_id": person.get("tracking_id"),
                "person_confidence": person["confidence"],
                "person_box": box,
                "head_box": head_box,
                "association_score": round(score, 3),
            })

        return motorcycles, persons

