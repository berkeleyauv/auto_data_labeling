import gradio as gr
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw
from utils import (
    describe_gate,
    draw_JSON_kpts,
    export_no_gate_pose,
    export_to_intro_detect,
    export_to_yolo,
    load_predictions,
    reviewed_stems,
)

def launch_qa(json_path,
    image_dir,
    server_port,
    export_format="pose",
    detect_dir=None,
    skip_truncated=False,
    images="none",
    resume=False,
):

    print(f"Reading JSON from {json_path}")
    print(f"Reading images from {image_dir}")
    print(f"Export format: {export_format}")

    raw_preds = load_predictions(json_path)

    img_fn = list(raw_preds.keys())

    labels_dir = image_dir.parent / "labels"
    detect_dir = Path(detect_dir) if detect_dir else image_dir.parent / "intro_detect"
    
    already_reviewed = 0
    if resume:
        done = reviewed_stems(labels_dir, detect_dir)
        remaining = [fn for fn in img_fn if Path(fn).stem not in done]
        already_reviewed = len(img_fn) - len(remaining)
        img_fn = remaining
        print(f"Resuming: {already_reviewed} images already reviewed, {len(img_fn)} left")
    do_pose = export_format in ("pose", "both")
    do_detect = export_format in ("detect", "both")

    
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
            gate_info = gr.Markdown()

        # Corner selections for adjustment
        with gr.Row():
            corner_selector = gr.Radio(
                choices=["TL", "TR", "BL", "BR"], 
                value="TL", 
                label="Corner to Adjust", 
                interactive=True
            )
            btn_remove = gr.Button("Remove Selected Corner")

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
            preview = describe_gate(raw_preds[img_fn[index]], skip_truncated)
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

        # Delete selected corner
        def remove_kp(index, corner):
            if index >= len(img_fn):
                return render_all(index)
 
            raw_preds[img_fn[index]][corner] = None
 
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
                    no_gate=no_gate, allow_truncated=not skip_truncated, images=images,
                )
                if row["status"] == "exported":
                    msg = f"Last: {filename} -> gate label written" + (" (truncated)" if row["truncated"] else "")
                elif row["status"] == "no_gate":
                    msg = f"Last: {filename} -> no gate (empty label)"
                else:
                    msg = f"Last: {filename} SKIPPED in detect export ({row['reason']})"
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

        app.load(fn=render_all, inputs=curr_idx, outputs=[img_display, gate_info])

        img_display.select(
            fn=update_kp,
            inputs=[curr_idx, corner_selector],
            outputs=[img_display, gate_info],
        )

        btn_remove.click(
            fn=remove_kp,
            inputs=[curr_idx, corner_selector],
            outputs=[img_display, gate_info],
        )


        btn_accept.click(
            fn=accept_and_next,
            inputs=curr_idx,
            outputs=[curr_idx, img_display, gate_info],
        )

        btn_no_gate.click(
            fn=no_gate_and_next,
            inputs=curr_idx,
            outputs=[curr_idx, img_display, gate_info]
        )

    app.launch(server_name="0.0.0.0", server_port=server_port, share=True)

if __name__ == "__main__":

    # Setup paths for JSON and exported labels
    parser = argparse.ArgumentParser(description="YOLO Keypoint QA")
    parser.add_argument("--json_path", type=str, default="./Test_JSON/raw_predictions.json")
    parser.add_argument("--image_dir", type=str, default="./Test_Images")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--resume", action="store_true", help="skip images already reviewed in an earlier session (they have a label or a manifest row)")

    # Which labels to write: YOLO-pose (labels/), the intro project's YOLO-detect set (intro_detect/), or both
    parser.add_argument("--format", type=str, choices=["pose", "detect", "both"], default="pose")
    parser.add_argument("--detect_dir", type=str, default=None,
                        help="where the detect labels + manifest.csv go (default: <dataset>/intro_detect)")
    parser.add_argument("--skip_truncated", action="store_true",
                        help="skip single-post gates in BOTH exports instead of keeping them (box extends to the image edge)")
    parser.add_argument("--images", type=str, choices=["none", "copy", "link"], default="none",
                        help="detect export: also copy/symlink each image next to its label (the current intro-project `split` expects that)")
 
    args = parser.parse_args()
    launch_qa(
        Path(args.json_path),
        Path(args.image_dir),
        args.port,
        export_format=args.format,
        detect_dir=args.detect_dir,
        skip_truncated=args.skip_truncated,
        images=args.images,
        resume=args.resume,
    )
