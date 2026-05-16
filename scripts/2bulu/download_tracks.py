"""Download KML/GPX track files from 2bulu.com with auto captcha solving.

Usage:
    python scripts/2bulu/download_tracks.py --csv scripts/2bulu/track_urls.csv --out data/2bulu_kml/
"""

import argparse
import csv
import os
import random
import sys
import time
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import requests
from DrissionPage import ChromiumPage, ChromiumOptions
from DrissionPage.common import Actions

READY_FILE = os.path.join(os.path.dirname(__file__), ".ready_to_download")

# ---------------------------------------------------------------------------
# Slider captcha solver
# ---------------------------------------------------------------------------

def _fetch_image(url: str) -> bytes | None:
    """Fetch image from URL, handling both http and data: URLs."""
    if url.startswith("data:"):
        # data:image/png;base64,xxxxx
        try:
            base64_part = url.split(",", 1)[1]
            import base64
            return base64.b64decode(base64_part)
        except Exception:
            return None
    else:
        try:
            resp = requests.get(url, timeout=10)
            return resp.content
        except Exception:
            return None


def solve_slider_captcha(page, max_retries=3):
    """Solve Aliyun slider captcha on 2bulu download page.

    Returns True if captcha solved, False otherwise.
    """
    for attempt in range(max_retries):
        try:
            # The captcha iframe - try multiple possible IDs (original code has typos)
            iframe = None
            for iframe_id in [
                '#aliyunCaptcha-window-popup',
                '#aliyunCaptciha-window-popup',
                '#aliyunCaptcha-window-popaup',
            ]:
                try:
                    iframe = page(iframe_id)
                    if iframe:
                        break
                except Exception:
                    continue

            if not iframe:
                print(f"    Captcha iframe not found (attempt {attempt+1})")
                time.sleep(1)
                continue

            # Get background image URL and download it
            bg_img_elem = None
            for img_selector in [
                'x://*[@id="aliyunCaptcha-img"]',
                'x://*[@id="aliyunCapitcha-img"]',
                'x://*[@id="aliyunjCaptcha-img"]',
                'x://*[@id="aliyunCaeptcha-img"]',
            ]:
                try:
                    bg_img_elem = iframe.ele(img_selector)
                    if bg_img_elem:
                        break
                except Exception:
                    continue

            if not bg_img_elem:
                continue

            bg_url = bg_img_elem.link
            bg_data = _fetch_image(bg_url)
            if bg_data is None:
                continue
            bg_img = cv2.imdecode(np.frombuffer(bg_data, np.uint8), cv2.IMREAD_COLOR)

            # Get puzzle piece image
            puzzle_elem = None
            for puzzle_selector in [
                'x://*[@id="aliyunCaptcha-puzzele"]',
                'x://*[@id="aliyunjCaptcha-puzzele"]',
                'x://*[@id="aliyunCaptcha-puzzle"]',
            ]:
                try:
                    puzzle_elem = iframe.ele(puzzle_selector)
                    if puzzle_elem:
                        break
                except Exception:
                    continue

            if not puzzle_elem:
                continue

            puzzle_url = puzzle_elem.link
            puzzle_data = _fetch_image(puzzle_url)
            if puzzle_data is None:
                continue
            puzzle_img = cv2.imdecode(np.frombuffer(puzzle_data, np.uint8), cv2.IMREAD_COLOR)

            if bg_img is None or puzzle_img is None:
                continue

            # Find puzzle piece position using template matching
            gap_x = _find_gap_position(bg_img, puzzle_img)

            if gap_x is None or gap_x < 10:
                # Refresh captcha and retry
                _refresh_captcha(iframe)
                time.sleep(0.5)
                continue

            # Find the slider element
            slider = None
            for slider_selector in [
                'x://*[@id="aliyunCaptcha-sliding-slider"]',
                'x://*[@id="aliyunCaptcha-sliding-slaider"]',
            ]:
                try:
                    slider = iframe.ele(slider_selector)
                    if slider:
                        break
                except Exception:
                    continue

            if not slider:
                continue

            # Execute the drag
            _human_drag(page, slider, gap_x)

            time.sleep(1.5)

            # Check if captcha disappeared (solved)
            try:
                page('#aliyunCaptcha-window-popup')
                page('#aliyunCaptciha-window-popup')
                # Still present - captcha not solved
                print(f"    Captcha still present after drag (attempt {attempt+1})")
                continue
            except Exception:
                # Captcha gone = solved!
                return True

        except Exception as e:
            print(f"    Captcha error: {e} (attempt {attempt+1})")
            time.sleep(1)

    return False


def _find_gap_position(bg_img, puzzle_img):
    """Find x-position of the gap in background image using edge detection + template matching."""
    h, w = bg_img.shape[:2]

    # Method 1: Canny edge detection + template matching
    bg_gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
    puzzle_gray = cv2.cvtColor(puzzle_img, cv2.COLOR_BGR2GRAY)

    # Canny edge detection
    bg_edges = cv2.Canny(bg_gray, 50, 150)
    puzzle_edges = cv2.Canny(puzzle_gray, 50, 150)

    # Template matching on edges
    result = cv2.matchTemplate(bg_edges, puzzle_edges, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val > 0.3:
        return max_loc[0]

    # Method 2: Try on original grayscale
    result2 = cv2.matchTemplate(bg_gray, puzzle_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val2, _, max_loc2 = cv2.minMaxLoc(result2)

    if max_val2 > 0.4:
        return max_loc2[0]

    return None


def _refresh_captcha(iframe):
    """Click the refresh button on the captcha."""
    for refresh_selector in [
        'x://*[@id="aliyunCaptcha-btn-refresh"]',
        'x://*[@id="aliyunCapthcha-btn-refrgeesh"]',
        'x://*[@id="aliyunCaptcha-btn-referesh"]',
    ]:
        try:
            btn = iframe.ele(refresh_selector)
            btn.click()
            return
        except Exception:
            continue


def _human_drag(page, slider_elem, distance):
    """Simulate human-like dragging on the slider."""
    ac = Actions(page)

    # The slider position in the iframe vs actual drag distance may differ
    # Apply a scaling factor (determined empirically)
    drag_distance = distance * 0.95  # slight adjustment

    ac.hold(slider_elem)
    time.sleep(0.05)

    # Simulate human drag with varying speed
    moved = 0
    while moved < drag_distance:
        remaining = drag_distance - moved
        if remaining < 5:
            step = remaining
        elif remaining < 20:
            step = random.uniform(1, 3)
        else:
            step = random.uniform(3, 8)
        step = min(step, remaining)

        ac.right(int(step))
        moved += step
        time.sleep(random.uniform(0.005, 0.02))

    # Small overshoot and correction (human-like)
    time.sleep(random.uniform(0.05, 0.15))
    ac.right(random.randint(-2, 2))
    time.sleep(random.uniform(0.1, 0.3))

    ac.release()


# ---------------------------------------------------------------------------
# Main download logic
# ---------------------------------------------------------------------------

def wait_for_login(page: ChromiumPage, timeout: int = 600) -> bool:
    """Wait until user is logged in."""
    print(f"\n{'='*60}")
    print(f"Chrome opened. Please log in to 2bulu.com.")
    print(f"When ready, create this empty file:")
    print(f"  {READY_FILE}")
    print(f"{'='*60}\n")

    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(READY_FILE):
            os.remove(READY_FILE)
            print("Ready! Starting downloads...\n")
            return True

        # Also auto-detect login
        try:
            # Check if "登录" link is gone (replaced by user menu)
            login_link = page.ele('@text()=登录')
            if not login_link:
                # Check more carefully
                user_avatar = page.ele('@class=user-avatar') or page.ele('@class=avatar')
                if user_avatar:
                    print("Auto-detected login! Starting downloads...\n")
                    return True
        except Exception:
            pass

        elapsed = int(time.time() - t0)
        if elapsed % 15 == 0:
            mins, secs = divmod(elapsed, 60)
            print(f"[{mins:02d}:{secs:02d}] Waiting for login...")
        time.sleep(3)

    return False


def download_one_track(page: ChromiumPage, url: str, track_num: int = 0) -> str:
    """Download KML for a single track.

    Returns: 'ok', 'skip', or 'captcha_fail'
    """
    page.get(url)
    time.sleep(1.5)

    # Check if track was deleted / private
    try:
        none_elem = page.ele('x://*[@id="base_area"]/div/p/a')
        if none_elem and none_elem.text == '首页':
            return 'skip'
    except Exception:
        pass

    # Click "下载" button (the down-arrow icon in the action bar)
    try:
        download_btn = page.ele('x://*[@id="pointPannel"]/a')
        download_btn.click()
        time.sleep(0.3)
        downloads_menu = page.ele('x://*[@id="base_area"]/div[8]/ul/li[2]')
        downloads_menu.click()
    except Exception:
        # Try dismissing a warning popup first
        try:
            popup_ok = page.ele('x:/html/body/div[6]/div[4]/div/input')
            popup_ok.click()
            time.sleep(0.5)
            download_btn = page.ele('x://*[@id="pointPannel"]/a')
            download_btn.click()
            downloads_menu = page.ele('x://*[@id="base_area"]/div[8]/ul/li[2]')
            downloads_menu.click()
        except Exception:
            return 'skip'

    time.sleep(0.3)

    # Click "KML轨迹" option
    try:
        kml_btn = page.ele('x://*[@id="base_area"]/div[8]/div[3]/ul/li/p[1]')
        kml_btn.click()
        time.sleep(0.5)
    except Exception:
        return 'skip'

    # Now the Aliyun captcha should appear. Solve it.
    time.sleep(1)
    solved = solve_slider_captcha(page, max_retries=3)

    if solved:
        return 'ok'
    else:
        return 'captcha_fail'


def main():
    parser = argparse.ArgumentParser(description="Download 2bulu KML tracks")
    parser.add_argument("--csv", type=str, required=True, help="CSV with track URLs")
    parser.add_argument("--out", type=str, default="data/2bulu_kml", help="Output dir")
    parser.add_argument("--delay-min", type=float, default=3.0, help="Min delay between requests")
    parser.add_argument("--delay-max", type=float, default=6.0, help="Max delay between requests")
    parser.add_argument("--login-timeout", type=int, default=600, help="Seconds to wait for login")
    args = parser.parse_args()

    urls = []
    with open(args.csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get("标题链接", row.get("url", "")).strip()
            region = row.get("区域", row.get("region", "unknown")).strip()
            if url:
                urls.append((region, url))

    print(f"Loaded {len(urls)} track URLs")
    os.makedirs(args.out, exist_ok=True)

    out_abs = os.path.abspath(args.out)
    co = ChromiumOptions()
    co.set_pref("download.default_directory", out_abs)
    co.set_pref("download.prompt_for_download", False)

    page = ChromiumPage(co)
    print(f"Download dir: {out_abs}")

    page.get("https://www.2bulu.com/")

    if not wait_for_login(page, timeout=args.login_timeout):
        print("Login timed out. Exiting.")
        sys.exit(1)

    ok = skip = captcha_fail = 0
    total = len(urls)
    track_num = 0

    for i, (region, url) in enumerate(urls):
        print(f"\n[{i+1}/{total}] {region}")
        try:
            result = download_one_track(page, url, track_num)
            if result == 'ok':
                ok += 1
                track_num += 1
            elif result == 'captcha_fail':
                captcha_fail += 1
            else:
                skip += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            skip += 1

        # Rate limiting
        delay = random.uniform(args.delay_min, args.delay_max)
        if i < total - 1:
            time.sleep(delay)

        if (i + 1) % 8 == 0:
            rest = random.randint(15, 30)
            print(f"\n  --- Longer break {rest}s ---")
            time.sleep(rest)

    print(f"\n{'='*50}")
    print(f"DONE: {ok} ok, {captcha_fail} captcha-failed, {skip} skipped")
    print(f"Files in: {out_abs}")
    print(f"Close browser when ready.")


if __name__ == "__main__":
    main()
