"""Launch the mock UI or an explicitly selected deployment factory."""
from __future__ import annotations
import argparse
import importlib
import json
import os
import signal
from pathlib import Path
import time
from .adapters.mock import MockEmbodiment, MockPolicy
from .contracts import SessionConfig
from .session import DeploymentSession


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory",help="MODULE:FUNCTION returning (config, policy, embodiment)")
    parser.add_argument("--config")
    repo=Path(__file__).resolve().parents[2]
    default_output=os.environ.get("DEPLOYMENT_OUTPUT_ROOT",str(repo/".runtime/results"))
    parser.add_argument("--output-root",default=default_output)
    parser.add_argument("--target-hz",type=float,default=60)
    parser.add_argument("--max-steps",type=int,default=5000)
    parser.add_argument("--no-video",action="store_true")
    parser.add_argument("--reference-manifest")
    parser.add_argument("--headless-demo",action="store_true",help="CPU mock only; one automatic success without devices")
    parser.add_argument("--gui-smoke",action="store_true",help="CPU mock only; automatically exercise and close the GUI")
    parser.add_argument("--screenshot",help="Save one mock GUI smoke screenshot outside the checkout")
    args=parser.parse_args()
    if (args.headless_demo or args.gui_smoke) and args.factory:
        parser.error("Automated demo/smoke is allowed only with the mock adapter")
    if args.screenshot and not args.gui_smoke:
        parser.error("--screenshot requires --gui-smoke")
    if args.factory:
        module,function=args.factory.split(":",1)
        config,policy,embodiment=getattr(importlib.import_module(module),function)(args.config)
    else:
        config=SessionConfig(method="mock",output_root=args.output_root,target_hz=args.target_hz,
                             max_steps=args.max_steps,record_video=not args.no_video,
                             reference_manifest=args.reference_manifest)
        policy,embodiment=MockPolicy(),MockEmbodiment()
    if args.headless_demo:
        session=DeploymentSession(config,policy,embodiment,repo)
        session.start()
        deadline=time.monotonic()+10
        while session.state not in {"READY","ERROR"}:
            if time.monotonic()>deadline:
                raise TimeoutError("Mock startup timeout")
            time.sleep(.01)
        if session.state=="ERROR":
            raise RuntimeError(session.error)
        session.start_rollout()
        while len(session.rows)<5 and session.state!="ERROR":
            if time.monotonic()>deadline:
                raise TimeoutError("Mock rollout timeout")
            time.sleep(.01)
        session.signal("success")
        while session.state not in {"READY","ERROR"}:
            if time.monotonic()>deadline:
                raise TimeoutError("Mock outcome timeout")
            time.sleep(.01)
        session.close()
        session.worker.join(5)
        if session.worker.is_alive() or session.error:
            raise RuntimeError(session.error or "Mock shutdown timeout")
        print(json.dumps({"folder":str(session.folder),"results":session.results},indent=2))
        return
    # Create a usable desktop before connecting to policy or hardware.
    import tkinter as tk
    from .gui import DeploymentGUI
    root=tk.Tk()
    app=DeploymentGUI(root,DeploymentSession(config,policy,embodiment,repo))
    def shutdown(_number,_frame):
        root.after(0,app.close)
    for number in (signal.SIGINT,signal.SIGTERM):
        signal.signal(number,shutdown)
    if args.gui_smoke:
        def smoke():
            if app.session.state=="READY" and not app.session.results:
                app.start()
            if len(app.session.rows)>=15 and not app.session.results:
                if args.screenshot:
                    from PIL import ImageGrab
                    ImageGrab.grab(xdisplay=os.environ.get("DISPLAY")).save(args.screenshot)
                app.signal("success")
            if app.session.results and app.session.state=="READY":
                app.close()
                return
            root.after(50,smoke)
        root.after(100,smoke)
        root.after(10000,app.close)
    root.mainloop()
    if app.session.error:
        raise RuntimeError(app.session.error)


if __name__=="__main__":
    main()
