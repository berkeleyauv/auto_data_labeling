import gradio as gr
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw
from utils import (
    EDITABLE_PARTS,
    describe_gate,
    describe_parts,
    draw_JSON_kpts,
    export_no_gate_pose,
    export_to_enrich,
    export_to_intro_detect,
    export_to_yolo,
    load_parts,
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
    enrich=False,
    parts_path=None,
):

    print(f"Reading JSON from {json_path}")
    print(f"Reading images from {image_dir}")
    print(f"Export format: {'enrich (--format is ignored)' if enrich else export_format}")

    raw_preds = load_predictions(json_path)

    img_fn = list(raw_preds.keys())

    labels_dir = image_dir.parent / "labels"
    detect_dir = Path(detect_dir) if detect_dir else image_dir.parent / "intro_detect"
    
    # --enrich: the contiguous gate + parts label goes to its own folder, and pose/detect are NOT written
    enrich_dir = image_dir.parent / "labels_enrich"
    raw_parts = {}
    if enrich:
        parts_path = Path(parts_path) if parts_path else Path(json_path).parent / "raw_parts.json"
        if not parts_path.is_file():
            raise SystemExit(f"--enrich needs {parts_path} (written by annotate_dataset.py --enrich). Pass --parts_path if it is elsewhere.")
        print(f"Reading part boxes from {parts_path}")
        raw_parts = load_parts(parts_path, img_fn)

    already_reviewed = 0
    if resume:
        # enrich_dir has no manifest.csv, so passing it twice just means "has a label in enrich_dir"
        done = reviewed_stems(enrich_dir, enrich_dir) if enrich else reviewed_stems(labels_dir, detect_dir)
        remaining = [fn for fn in img_fn if Path(fn).stem not in done]
        already_reviewed = len(img_fn) - len(remaining)
        img_fn = remaining
        print(f"Resuming: {already_reviewed} images already reviewed, {len(img_fn)} left")
    do_pose = export_format in ("pose", "both")
    do_detect = export_format in ("detect", "both")

    
    do_pose = export_format in ("pose", "both")
    do_detect = export_format in ("detect", "both")
    do_enrich = enrich
    if enrich:
        do_pose = do_detect = False

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

        # --enrich: drop a part box that is wrong (it is then exported as 0 0 0 0)
        if enrich:
            with gr.Row():
                part_selector = gr.Radio(
                    choices=list(EDITABLE_PARTS),
                    value=EDITABLE_PARTS[0],
                    label="Part box",
                    interactive=True
                )
                btn_remove_part = gr.Button("Remove Selected Part")

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
            return draw_JSON_kpts(img_path, kpts, raw_parts.get(filename) if enrich else None)

        # Text shown under image to inform last export results + preview of orientation
        def status_text(index):
            if index >= len(img_fn):
                return last_msg[0] or "All images reviewed."
            preview = describe_gate(raw_preds[img_fn[index]], skip_truncated)
            if enrich:
                preview += "\n\n" + describe_parts(raw_parts[img_fn[index]])
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

        # Delete the selected part box (enrich only)
        def remove_part(index, part):
            if index >= len(img_fn):
                return render_all(index)

            raw_parts[img_fn[index]][part] = None

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
            if do_enrich:
                status = export_to_enrich(filename, points, raw_parts[filename], img_width, img_height,
                                          enrich_dir, no_gate=no_gate, allow_truncated=not skip_truncated)
                msg = {"exported": f"Last: {filename} -> enrich label written",
                       "no_gate": f"Last: {filename} -> no gate (empty label)",
                       "skipped": f"Last: {filename} SKIPPED in enrich export (corners form no valid box)"}[status]
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


        if enrich:
            btn_remove_part.click(
                fn=remove_part,
                inputs=[curr_idx, part_selector],
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
 
    # Not a standard YOLO format: gate box + corners + left post / middle divider / right post / crossbar boxes on one line
    parser.add_argument("--enrich", action="store_true",
                        help="write the contiguous enrich label to <dataset>/labels_enrich/ INSTEAD of pose/detect (needs raw_parts.json from annotate_dataset.py --enrich)")
    parser.add_argument("--parts_path", type=str, default=None,
                        help="--enrich: path to raw_parts.json (default: next to --json_path)")

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
        enrich=args.enrich,
        parts_path=args.parts_path,
    )