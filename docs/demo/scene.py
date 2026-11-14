"""Demo scene for the Streamlit viewer. Regenerate:
  1. mlia seed && mlia train && mlia run --provider fixtures --llm fake   (offline, fictional data;
     point MLIA_DATABASE_URL / MLIA_OUTPUT_DIR at a scratch location)
  2. streamlit run app/streamlit_app.py --server.port 8740
  3. python docs/demo/scene.py
  4. python ~/.claude/skills/demo-gif/scripts/compose.py docs/demo/build --out docs/demo.gif
"""
import os, sys; sys.path.insert(0, os.path.expanduser("~/.claude/skills/demo-gif/scripts"))
from pathlib import Path
from director import Director, run

URL = "http://localhost:8740"


async def scene(d: Director):
    p = d.page
    # warm the server (imports, cache) in a throwaway context so the recorded load is clean
    warm = await p.context.browser.new_context()
    wp = await warm.new_page()
    await wp.goto(URL)
    await wp.get_by_role("button", name="Download Excel").wait_for(timeout=120_000)
    await warm.close()

    await d.goto(URL, settle=2)
    await p.get_by_role("button", name="Download Excel").wait_for(timeout=60_000)
    await d.caption("Open the latest run", step=1)
    await d.hold(1.2)

    await d.caption("Keep only high-intent rows", step=2)
    await d.scroll(170)
    box = await p.locator('[data-testid="stSlider"]').first.bounding_box()
    y = box["y"] + box["height"] - 14
    thumb_y = (await p.get_by_role("slider").first.bounding_box())
    y = thumb_y["y"] + thumb_y["height"] / 2
    await d.click((box["x"] + 6 + 0.70 * (box["width"] - 12) - 6, y - 6, 12, 12))
    await p.wait_for_timeout(1500)

    await d.caption("Read the score and why", step=3)
    grid = p.locator('[data-testid="stDataFrame"]').first
    await d.focus(grid, zoom=1.35)
    await d.hold(1.8)
    await d.unfocus()

    await d.caption("Open the market digest", step=4)
    await d.click(p.get_by_text("Market digest").first)
    await d.hold(1.6)

    await d.caption("Give broker feedback", step=5)
    await d.click(p.get_by_role("button", name="Save"))
    await d.hold(1.8)


run(scene, out=str(Path(__file__).with_name("build")), title="Market & Lead Intelligence Agent",
    subtitle="From public market signals to scored, broker-ready leads", url_label="localhost:8740")
