from microlane.schemas.prediction import Prediction
from microlane.evaluation.core.lane_iou import LaneIoU


def calculate_iou(prediction: Prediction) -> float:
    gt_sample  = prediction.samples[-1]
    gt_lanes   = gt_sample.lanes
    pred_lanes = prediction.lanes
    y_samples  = prediction.h_samples
    return LaneIoU.compute_iou(pred_lanes, gt_lanes, y_samples)


# def calculate_iou(prediction: Prediction, visualize: bool = True) -> float:

#     gt_sample  = prediction.samples[-1]
#     gt_lanes   = gt_sample.lanes
#     pred_lanes = prediction.lanes
#     y_samples  = prediction.h_samples

#     if visualize:
#         _visualize_iou(pred_lanes, gt_lanes, y_samples)

#     return LaneIoU.compute_iou(pred_lanes, gt_lanes, y_samples)


# def _visualize_iou(pred_lanes, gt_lanes, y_samples) -> None:
#     """Debug visualization — call via calculate_iou(..., visualize=True)."""
#     import matplotlib.pyplot as plt
#     from itertools import combinations

#     fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0f0f12")
#     fig.suptitle("Lane IoU debug", color="white", fontsize=13)

#     for ax, lanes, label, color in [
#         (axes[0], gt_lanes,   "GT lanes",   "#34c382"),
#         (axes[1], pred_lanes, "Pred lanes", "#6394ff"),
#     ]:
#         ax.set_facecolor("#0f0f12")
#         ax.tick_params(colors="gray")
#         for spine in ax.spines.values():
#             spine.set_edgecolor("#333")

#         best_poly, best_area, best_idx = None, 0.0, (-1, -1)

#         for i, j in combinations(range(len(lanes)), 2):
#             poly = LaneIoU.build_polygon(lanes[i], lanes[j], y_samples)
#             if poly is None:
#                 continue
#             xs, ys = poly.exterior.xy
#             ax.fill(xs, ys, alpha=0.08, color=color)
#             ax.plot(xs, ys, lw=0.6, alpha=0.25, color=color, linestyle="--")
#             if poly.area > best_area:
#                 best_area, best_poly, best_idx = poly.area, poly, (i, j)

#         for i, lane in enumerate(lanes):
#             pts = LaneIoU.lane_to_points(lane, y_samples)
#             if not pts:
#                 continue
#             lx, ly = zip(*pts)
#             is_best = i in best_idx
#             ax.plot(
#                 lx, ly,
#                 lw=2.5 if is_best else 1.0,
#                 color=color if is_best else "#888",
#                 alpha=1.0 if is_best else 0.4,
#                 marker="o", markersize=3,
#                 label=f"lane {i}" + (" ★" if is_best else ""),
#             )

#         if best_poly:
#             xs, ys = best_poly.exterior.xy
#             ax.fill(xs, ys, alpha=0.25, color=color)
#             ax.plot(xs, ys, lw=1.5, color=color)

#         ax.set_title(f"{label}  (best area={best_area:.0f}px²)", color="white", fontsize=11)
#         ax.invert_yaxis()
#         ax.legend(fontsize=8, loc="lower right",
#                   facecolor="#1a1a22", labelcolor="white", edgecolor="#333")
#         ax.set_xlabel("x", color="gray")
#         ax.set_ylabel("y", color="gray")

#     gt_poly   = LaneIoU.best_pair_polygon(gt_lanes,   y_samples)
#     pred_poly = LaneIoU.best_pair_polygon(pred_lanes, y_samples)

#     if gt_poly and pred_poly:
#         fig2, ax2 = plt.subplots(figsize=(8, 6), facecolor="#0f0f12")
#         ax2.set_facecolor("#0f0f12")
#         ax2.tick_params(colors="gray")
#         for spine in ax2.spines.values():
#             spine.set_edgecolor("#333")

#         for poly, c, ls, lbl in [
#             (gt_poly,   "#34c382", "-",  "GT poly"),
#             (pred_poly, "#6394ff", "--", "Pred poly"),
#         ]:
#             xs, ys = poly.exterior.xy
#             ax2.fill(xs, ys, alpha=0.2, color=c)
#             ax2.plot(xs, ys, lw=1.5, color=c, linestyle=ls, label=lbl)

#         inter = gt_poly.intersection(pred_poly)
#         union = gt_poly.union(pred_poly)
#         iou   = inter.area / union.area if union.area else 0.0

#         if inter.area > 0:
#             xs, ys = inter.exterior.xy
#             ax2.fill(xs, ys, alpha=0.55, color="#ffc832",
#                      label=f"Intersection ({inter.area:.0f}px²)")
#             ax2.plot(xs, ys, lw=1.2, color="#ffc832")

#         uxs, uys = union.exterior.xy
#         ax2.plot(uxs, uys, lw=1.0, color="white", alpha=0.2, linestyle=":",
#                  label=f"Union ({union.area:.0f}px²)")

#         ax2.set_title(f"IoU overlap  —  IoU = {iou:.4f}", color="white", fontsize=12)
#         ax2.invert_yaxis()
#         ax2.legend(fontsize=9, facecolor="#1a1a22", labelcolor="white", edgecolor="#333")
#         ax2.set_xlabel("x", color="gray")
#         ax2.set_ylabel("y", color="gray")
#         fig2.tight_layout()

#     fig.tight_layout()
#     plt.show()