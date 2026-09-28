import os
import json

import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QProgressDialog,
    QApplication,
    QInputDialog,
)


def _read_image(path):
    """
    支持 Windows 中文路径读取
    """
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _get_color(group_id):
    """
    根据真实 Track ID 生成稳定颜色。

    注意：
    这里仍然使用真实 group_id，
    所以同一个目标跨帧颜色始终保持一致。
    """

    if group_id is None:
        group_id = 0

    try:
        gid = int(group_id)
    except (TypeError, ValueError):
        gid = abs(hash(str(group_id))) % 10000

    return (
        int((37 * gid + 80) % 255),
        int((97 * gid + 140) % 255),
        int((17 * gid + 220) % 255),
    )


def _get_display_track_id(
    label,
    group_id,
    class_track_id_map,
    class_next_id,
):
    """
    将真实全局 Track ID 转换为“类别内显示 ID”。

    例如真实 ID：
        car -> 3
        person -> 4
        car -> 8

    视频显示：
        car-1
        person-1
        car-2

    注意：
    这里只改变视频显示编号，
    不修改 JSON 中真实 group_id。
    """

    if group_id is None:
        return None

    try:
        real_track_id = int(group_id)
    except (TypeError, ValueError):
        real_track_id = str(group_id)

    # 第一次遇到这个类别
    if label not in class_track_id_map:
        class_track_id_map[label] = {}
        class_next_id[label] = 1

    # 第一次遇到这个类别下的真实 Track ID
    if real_track_id not in class_track_id_map[label]:

        display_id = class_next_id[label]

        class_track_id_map[label][real_track_id] = display_id

        class_next_id[label] += 1

    return class_track_id_map[label][real_track_id]


def _draw_shape(
    frame,
    shape,
    class_track_id_map,
    class_next_id,
):
    """
    将 X-AnyLabeling JSON 中的 Shape 绘制到视频帧。

    视频中的 Track ID 按类别分别编号：
        car-1
        car-2
        person-1
        person-2
        bus-1
        ...
    """

    label = str(shape.get("label", "object"))

    # X-AnyLabeling / Tracker 中真正的全局 ID
    group_id = shape.get("group_id", None)

    score = shape.get("score", None)

    shape_type = shape.get("shape_type", "polygon")

    points = shape.get("points", [])

    if not points:
        return frame

    # ---------------------------------------------
    # 颜色仍然使用真实 Track ID
    # 同一目标跨帧颜色不会改变
    # ---------------------------------------------
    color = _get_color(group_id)

    # =============================================
    # Polygon
    # =============================================

    if shape_type == "polygon":

        pts = np.array(points, dtype=np.int32)

        if len(pts) < 3:
            return frame

        # 半透明 Mask
        overlay = frame.copy()

        cv2.fillPoly(
            overlay,
            [pts],
            color,
        )

        frame = cv2.addWeighted(
            overlay,
            0.35,
            frame,
            0.65,
            0,
        )

        # Polygon 边界
        cv2.polylines(
            frame,
            [pts],
            True,
            color,
            2,
        )

        x = int(np.min(pts[:, 0]))
        y = int(np.min(pts[:, 1]))

    # =============================================
    # Rectangle
    # =============================================

    elif shape_type == "rectangle" and len(points) >= 2:

        x1, y1 = map(int, points[0])
        x2, y2 = map(int, points[1])

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        x, y = x1, y1

    else:
        return frame

    # =============================================
    # 获取“类别内显示 ID”
    # =============================================

    display_id = _get_display_track_id(
        label,
        group_id,
        class_track_id_map,
        class_next_id,
    )

    # =============================================
    # 标签文字
    # =============================================

    if display_id is not None:

        # 例如：
        # car-1
        # car-2
        # person-1
        text = f"{label}-{display_id}"

    else:

        # 没有 Track ID 的目标
        text = label

    # 置信度
    if score is not None:
        try:
            text += f" {float(score):.2f}"
        except (TypeError, ValueError):
            pass

    # 防止文字超出顶部
    y = max(y - 8, 20)

    # =============================================
    # 文字背景
    # =============================================

    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        2,
    )

    box_x1 = x
    box_y1 = max(
        y - text_height - baseline - 4,
        0,
    )

    box_x2 = x + text_width + 6
    box_y2 = y + baseline + 2

    cv2.rectangle(
        frame,
        (box_x1, box_y1),
        (box_x2, box_y2),
        color,
        -1,
    )

    # =============================================
    # 标签文字
    # =============================================

    cv2.putText(
        frame,
        text,
        (x + 3, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return frame


def export_annotated_video(self):
    """
    将当前 X-AnyLabeling 工作区中的
    图片 + JSON 标注
    合成为带实例分割结果和追踪 ID 的视频。

    视频显示 ID 按类别单独编号，
    但 JSON 中真实 group_id 保持不变。
    """

    # =============================================
    # 1. 当前图片列表
    # =============================================

    image_files = (
        list(self.image_list)
        if self.image_list
        else [self.filename]
    )

    image_files = [
        f
        for f in image_files
        if f and os.path.isfile(f)
    ]

    if not image_files:

        QMessageBox.warning(
            self,
            "导出失败",
            "当前没有可以导出的视频帧。",
        )

        return

    # ---------------------------------------------
    # 视频帧按文件名排序
    # frame_00000
    # frame_00001
    # frame_00002
    # ---------------------------------------------

    image_files.sort()

    # =============================================
    # 2. 选择原始视频
    # 用于读取原 FPS
    # =============================================

    source_video, _ = QFileDialog.getOpenFileName(
        self,
        "选择原始视频",
        "",
        "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv)",
    )

    if not source_video:
        return

    cap = cv2.VideoCapture(source_video)

    if not cap.isOpened():

        QMessageBox.warning(
            self,
            "读取失败",
            "无法读取原始视频。",
        )

        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    if original_fps <= 0:
        original_fps = 25.0

    # =============================================
    # 3. 输入当初的抽帧间隔
    # =============================================

    interval, ok = QInputDialog.getInt(
        self,
        "抽帧间隔",
        "请输入打开视频时使用的抽帧间隔：",
        1,
        1,
        1000,
        1,
    )

    if not ok:
        return

    output_fps = original_fps / interval

    # =============================================
    # 4. 选择保存位置
    # =============================================

    output_path, _ = QFileDialog.getSaveFileName(
        self,
        "保存标注视频",
        "annotated_video.mp4",
        "MP4 Video (*.mp4)",
    )

    if not output_path:
        return

    if not output_path.lower().endswith(".mp4"):
        output_path += ".mp4"

    # =============================================
    # 5. 第一帧确定尺寸
    # =============================================

    first_frame = _read_image(
        image_files[0]
    )

    if first_frame is None:

        QMessageBox.warning(
            self,
            "导出失败",
            "无法读取第一帧。",
        )

        return

    height, width = first_frame.shape[:2]

    # =============================================
    # 6. 创建 VideoWriter
    # =============================================

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        output_fps,
        (width, height),
    )

    if not writer.isOpened():

        QMessageBox.warning(
            self,
            "导出失败",
            "无法创建 MP4 文件。\n"
            "建议先使用纯英文保存路径。",
        )

        return

    # =============================================
    # 7. Label JSON 所在目录
    # =============================================

    label_dir = (
        self.output_dir
        if self.output_dir
        else os.path.dirname(image_files[0])
    )

    # =============================================
    # 8. 类别内 Track ID 映射
    # =============================================
    #
    # 真实：
    # car group_id=1
    # person group_id=2
    # car group_id=5
    #
    # 显示：
    # car-1
    # person-1
    # car-2
    #
    # 这两个变量必须放在帧循环外面！
    # =============================================

    class_track_id_map = {}

    class_next_id = {}

    # =============================================
    # 9. 进度窗口
    # =============================================

    progress = QProgressDialog(
        "正在生成标注视频...",
        "取消",
        0,
        len(image_files),
        self,
    )

    progress.setWindowTitle(
        "导出标注视频"
    )

    progress.setMinimumWidth(450)

    progress.show()

    # =============================================
    # 10. 每一帧绘制标注
    # =============================================

    written = 0

    for index, image_path in enumerate(
        image_files
    ):

        if progress.wasCanceled():
            break

        frame = _read_image(
            image_path
        )

        if frame is None:
            continue

        # -----------------------------------------
        # 尺寸不统一时自动调整
        # -----------------------------------------

        if (
            frame.shape[1] != width
            or frame.shape[0] != height
        ):

            frame = cv2.resize(
                frame,
                (width, height),
            )

        base_name = os.path.splitext(
            os.path.basename(image_path)
        )[0]

        json_path = os.path.join(
            label_dir,
            base_name + ".json",
        )

        # =========================================
        # 读取 JSON
        # =========================================

        if os.path.isfile(json_path):

            try:

                with open(
                    json_path,
                    "r",
                    encoding="utf-8",
                ) as f:

                    data = json.load(f)

                shapes = data.get(
                    "shapes",
                    [],
                )

                for shape in shapes:

                    frame = _draw_shape(
                        frame,
                        shape,
                        class_track_id_map,
                        class_next_id,
                    )

            except Exception as e:

                print(
                    f"读取标注失败: {json_path}",
                    e,
                )

        # =========================================
        # 写视频
        # =========================================

        writer.write(frame)

        written += 1

        progress.setValue(
            index + 1
        )

        QApplication.processEvents()

    # =============================================
    # 11. 完成
    # =============================================

    writer.release()

    progress.close()

    # ---------------------------------------------
    # 输出一下本次类别编号统计
    # ---------------------------------------------

    summary_lines = []

    for label in sorted(
        class_track_id_map.keys()
    ):

        count = len(
            class_track_id_map[label]
        )

        summary_lines.append(
            f"{label}: {count}"
        )

    if summary_lines:

        summary_text = "\n".join(
            summary_lines
        )

    else:

        summary_text = "未发现 Track ID"

    QMessageBox.information(
        self,
        "导出完成",
        (
            f"标注视频生成完成！\n\n"
            f"输出帧数：{written}\n"
            f"FPS：{output_fps:.2f}\n\n"
            f"追踪目标统计：\n"
            f"{summary_text}\n\n"
            f"保存位置：\n"
            f"{output_path}"
        ),
    )