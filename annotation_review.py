import gradio as gr
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw
from utils import (
    DEFAULT_CLEAR_RATIO,
    DEFAULT_HEAD_ON_RATIO,
    describe_orientation,
    draw_JSON_kpts,
    export_no_gate_pose,
    export_to_intro_detect,
    export_to_yolo,
    load_predictions,
)

def launch_qa(json_path,
    image_dir,
    server_port,
    export_format="pose",
    detect_dir=None,
    head_on_ratio=DEFAULT_HEAD_ON_RATIO,
    clear_ratio=DEFAULT_CLEAR_RATIO,
    skip_truncated=False,
    link_images=False,
):

    print(f"Reading JSON from {json_path}")
    print(f"Reading images from {image_dir}")
    print(f"Export format: {export_format}")

    raw_preds = load_predictions(json_path)

    img_fn = list(raw_preds.keys())

    labels_dir = image_dir.parent / "labels"
    detect_dir = Path(detect_dir) if detect_dir else image_dir.parent / "intro_detect"
    do_pose = export_format in ("pose", "both")
    do_detect = export_format in ("detect", "both")

    # Result of the most recent export, shown under the image
    last_msg = [""]

    with gr.Blocks(title="Underwater Gate Pose Annotator") as app:

        # State variable to keep track of index
        curr_idx = gr.State(0)

        # Image display
        with gr.Row():
            img_display = gr.Image(type="pil", interactive=True, label="Current Gate Frame")

        # Preivew assigned orientation class
        with gr.Row():
            orientation_info = gr.Markdown()

        # Corner selections for adjustment
        with gr.Row():
            corner_selector = gr.Radio(
                choices=["TL", "TR", "BL", "BR"], 
                value="TL", 
                label="Corner to Adjust", 
                interactive=True
            )

        # Accept button
        with gr.Row():
            btn_accept = gr.Button("Accept & Export")
            btn_no_gate = gr.Button("No Gate (export empty label)")

        # ---------------------------------------
        #           HELPER FUNCTIONS
        # ---------------------------------------

        # Display each set of predictions on its corresponding image
        def render_image(index):
            if index >= len(img_fn):
                return Image.open("alldone.png").convert("RGB")
            filename = img_fn[index]
            img_path = image_dir / filename
            kpts = raw_preds[filename]
            return draw_JSON_kpts(img_path, kpts)

        # Text shown under image to inform last export results + preview of orientation
        def status_text(index):
            if index >= len(img_fn):
                return last_msg[0] or "All images reviewed."
            preview = describe_orientation(raw_preds[img_fn[index]], head_on_ratio, clear_ratio)
            return f"{last_msg[0]}\n\n{preview}" if last_msg[0] else preview

        def render_all(index):
            return render_image(index), status_text(index)

        # Logic for updating keypoints
        def update_kp(index, corner, evt: gr.SelectData):
            if index >= len(img_fn):
                return render_all(index)
            
            x, y = evt.index

            filename = img_fn[index]

            raw_preds[filename][corner] = [x, y]

            return render_all(index)

        # Write whichever exports were requested for the current image
        def export_current(index, no_gate=False):
            filename = img_fn[index]
            points = raw_preds[filename]
 
            img_path = image_dir / filename
            with Image.open(img_path) as img:
                img_width, img_height = img.size
 
            if do_pose:
                if no_gate:
                    export_no_gate_pose(filename, labels_dir)
                else:
                    export_to_yolo(filename, points, img_width, img_height, labels_dir,
                                   allow_truncated=not skip_truncated)
 
            msg = f"Last: {filename} saved"
            if do_detect:
                row = export_to_intro_detect(
                    filename, points, img_path, img_width, img_height, detect_dir,
                    head_on_ratio, clear_ratio, no_gate=no_gate, link_images=link_images,
                )
                if row["status"] == "exported":
                    msg = f"Last: {filename} -> {row['class_name']} (class {row['class_id']}, ratio {row['ratio']})"
                elif row["status"] == "no_gate":
                    msg = f"Last: {filename} -> no gate (empty label)"
                else:
                    msg = f"Last: {filename} SKIPPED in intro export ({row['reason']})"
            last_msg[0] = msg

        # Accepting the corner points
        def accept_and_next(index):
            if index >= len(img_fn):
                return index, render_image(index), status_text(index)

            export_current(index)            

            # Move onto next image
            new_index = index + 1
            return new_index, render_image(new_index), status_text(new_index)

        # Reviewer confirms no gate
        def no_gate_and_next(index):
            if index >= len(img_fn):
                return index, render_image(index), status_text(index)
 
            export_current(index, no_gate=True)
 
            new_index = index + 1
            return new_index, render_image(new_index), status_text(new_index)

        # ---------------------------------------
        #            BUTTON CONFIGS
        # ---------------------------------------

        app.load(fn=render_all, inputs=curr_idx, outputs=[img_display, orientation_info])

        img_display.select(
            fn=update_kp,
            inputs=[curr_idx, corner_selector],
            outputs=[img_display, orientation_info],
        )

        btn_accept.click(
            fn=accept_and_next,
            inputs=curr_idx,
            outputs=[curr_idx, img_display, orientation_info],
        )

        btn_no_gate.click(
            fn=no_gate_and_next,
            inputs=curr_idx,
            outputs=[curr_idx, img_display, orientation_info]
        )

    app.launch(server_name="0.0.0.0", server_port=server_port, share=True)

if __name__ == "__main__":

    # Setup paths for JSON and exported labels
    parser = argparse.ArgumentParser(description="YOLO Keypoint QA")
    parser.add_argument("--json_path", type=str, default="./Test_JSON/raw_predictions.json")
    parser.add_argument("--image_dir", type=str, default="./Test_Images")
    parser.add_argument("--port", type=int, default=7860)

     # Which labels to write: YOLO-pose (labels/), the intro project's YOLO-detect set (intro_detect/), or both
    parser.add_argument("--format", type=str, choices=["pose", "detect", "both"], default="pose")
    parser.add_argument("--detect_dir", type=str, default=None,
                        help="where the intro-project dataset goes (default: <dataset>/intro_detect)")
    parser.add_argument("--head_on_ratio", type=float, default=DEFAULT_HEAD_ON_RATIO,
                        help="edge ratios within [1/x, x] are head-on")
    parser.add_argument("--clear_ratio", type=float, default=DEFAULT_CLEAR_RATIO,
                        help="edge ratios beyond x (or below 1/x) are clearly left/right; in between is skipped")
    parser.add_argument("--skip_truncated", action="store_true",
                        help="pose export: skip single-post gates instead of keeping them with v=0 corners")
    parser.add_argument("--link_images", action="store_true",
                        help="intro export: symlink images instead of copying them (saves disk)")
 
    args = parser.parse_args()
    launch_qa(
        Path(args.json_path),
        Path(args.image_dir),
        args.port,
        export_format=args.format,
        detect_dir=args.detect_dir,
        head_on_ratio=args.head_on_ratio,
        clear_ratio=args.clear_ratio,
        skip_truncated=args.skip_truncated,
        link_images=args.link_images,
    )
