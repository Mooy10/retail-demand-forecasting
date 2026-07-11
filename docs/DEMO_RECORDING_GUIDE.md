# Demo Recording Guide

Goal: record a 20-30 second silent product walkthrough for GitHub, LinkedIn, or a portfolio page.

## Suggested Flow

1. Start the dashboard:
   ```powershell
   streamlit run dashboard\app.py
   ```
2. Open `http://localhost:8501`.
3. Record only the browser window.
4. Move through these pages:
   - Home
   - Executive Overview
   - Demand Forecast
   - Inventory Planning
   - Model Performance
   - Business Insights
5. Show one interaction:
   - Select a state filter, or
   - Change inventory scenario from `base` to `conservative`.
6. Stop recording after the dashboard has shown the business recommendation page.

## Tools

### OBS Studio

- Add a Window Capture source for the browser.
- Set output to MP4 or MKV.
- Use 1920x1080 if possible.
- Record at 30 FPS.

### Xbox Game Bar

- Open the dashboard browser window.
- Press `Win + G`.
- Use Capture > Start Recording.
- Stop after the final page.

### PowerPoint or Clipchamp

- Use screen recording mode.
- Crop to the browser window.
- Export as MP4.

## Quality Checklist

- No credentials or local folders visible.
- No terminal output visible.
- No personal notifications visible.
- Dashboard warning about simulated inventory remains visible where relevant.
- Video length stays under 30 seconds for quick portfolio viewing.
