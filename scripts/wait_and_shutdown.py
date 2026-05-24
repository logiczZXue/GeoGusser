"""Wait for E2E test to finish, then shutdown. Run in background."""
import subprocess, time, os

# Save results when test is done
result_src = r"D:\Geocomp\output\regression\fusion_e2e_thinking_partial.json"
result_dst = r"D:\Geocomp\output\regression\fusion_e2e_results_thinking_v3.json"

print("[Watcher] Monitoring for test completion...")

while True:
    # Check if any python.exe is running the test
    result = subprocess.run(
        ['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'],
        capture_output=True, text=True
    )
    # If no python process (or only this script), test is done
    python_lines = [l for l in result.stdout.split('\n') if 'python.exe' in l.lower()]
    if len(python_lines) <= 1:  # Just header or nothing
        print("[Watcher] Python process finished. Saving results...")
        # Copy result file
        if os.path.exists(result_src):
            import shutil
            shutil.copy(result_src, result_dst)
            print(f"[Watcher] Results saved to {result_dst}")
        else:
            print(f"[Watcher] WARNING: {result_src} not found")

        print("[Watcher] Shutting down in 60 seconds...")
        subprocess.run(['C:/Windows/System32/shutdown.exe', '/s', '/f', '/t', '60'], shell=True)
        break

    time.sleep(30)  # Check every 30 seconds
