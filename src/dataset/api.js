/**
 * API Service Module
 * 
 * Uses Express and Puppeteer to provide 11 API endpoints that simulate Street View operations.
 */

import express from 'express';
import puppeteer from 'puppeteer';

const app = express();
const API_PORT = 2; // Default fallback port, will be dynamically modified by the Python wrapper

let browser;
let page;

/**
 * Delay function that returns a Promise for a given number of milliseconds
 * @param {number} ms - The number of milliseconds to delay
 */
async function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Launch Puppeteer and open the local static page
(async () => {
  try {
    // For debugging purposes, headless is set to false; change to true in production
    browser = await puppeteer.launch({ headless: false });
    page = await browser.newPage();
    // Load the static page (note: ensure that the app.js service is running)
    await page.goto('http://localhost:6657', { waitUntil: 'networkidle2' });
    await page.setViewport({ width: 1280, height: 800 });
    // Wait for the Street View container to load completely
    await page.waitForSelector('#street-view');
    // Delay to allow map data to fully load
    await delay(5000);
    console.log('Puppeteer page loaded successfully');

    // Automatically focus the Street View container
    await page.evaluate(() => {
      document.getElementById('street-view').focus();
    });

    // Simulate keyboard actions: press Tab twice then Enter
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');
    await page.keyboard.press('Enter');
  } catch (err) {
    console.error('Error launching Puppeteer:', err);
  }
})();

/**
 * Screenshot function: captures the current page and returns PNG data via the response
 * @param {Object} res - Express's response object
 */
async function captureScreenshot(res) {
  try {
    let screenshotBuffer = await page.screenshot({ type: 'png' });
    
    // Ensure screenshotBuffer is of type Buffer
    if (!Buffer.isBuffer(screenshotBuffer)) {
      if (screenshotBuffer.data) {
        screenshotBuffer = Buffer.from(screenshotBuffer.data);
      } else {
        screenshotBuffer = Buffer.from(screenshotBuffer);
      }
    }
    
    // Disable client caching
    res.set('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
    res.set('Pragma', 'no-cache');
    res.set('Expires', '0');
    res.removeHeader('ETag');
    
    // Set response header to image/png
    res.setHeader('Content-Type', 'image/png');
    
    // Send the PNG binary data
    res.end(screenshotBuffer, 'binary');
  } catch (err) {
    res.status(500).send(err.toString());
  }
}

// Define API endpoints

app.get('/api/get_status', async (req, res) => {
  await delay(5000);
  await captureScreenshot(res);
});

app.get('/api/look_left', async (req, res) => {
  try {
    await page.keyboard.press('j');
    await delay(100);
    res.send('Executed left turn operation');
  } catch (err) {
    res.status(500).send(err.toString());
  }
});

app.get('/api/look_right', async (req, res) => {
  try {
    await page.keyboard.press('l');
    await delay(100);
    res.send('Executed right turn operation');
  } catch (err) {
    res.status(500).send(err.toString());
  }
});

app.get('/api/look_up', async (req, res) => {
  try {
    await page.keyboard.press('i');
    await delay(100);
    res.send('Executed upward tilt operation');
  } catch (err) {
    res.status(500).send(err.toString());
  }
});

app.get('/api/look_down', async (req, res) => {
  try {
    await page.keyboard.press('k');
    await delay(100);
    res.send('Executed downward tilt operation');
  } catch (err) {
    res.status(500).send(err.toString());
  }
});

app.get('/api/forward', async (req, res) => {
  try {
    await page.keyboard.press('w');
    await delay(100);
    res.send('Executed forward movement');
  } catch (err) {
    res.status(500).send(err.toString());
  }
});

app.get('/api/backward', async (req, res) => {
  try {
    await page.keyboard.press('s');
    await delay(100);
    res.send('Executed backward movement');
  } catch (err) {
    res.status(500).send(err.toString());
  }
});

app.get('/api/turn_left', async (req, res) => {
  try {
    await page.keyboard.press('a');
    await delay(100);
    res.send('Executed left turn');
  } catch (err) {
    res.status(500).send(err.toString());
  }
});

app.get('/api/turn_right', async (req, res) => {
  try {
    await page.keyboard.press('d');
    await delay(100);
    res.send('Executed right turn');
  } catch (err) {
    res.status(500).send(err.toString());
  }
});

app.get('/api/zoom_in', async (req, res) => {
  try {
    const viewport = page.viewport();
    const centerX = viewport.width / 2;
    const centerY = viewport.height / 2;
    // Click the center of the page to ensure focus
    await page.mouse.click(centerX, centerY);
    // Simulate mouse wheel scrolling up for zoom in
    await page.mouse.wheel({ deltaY: -100 });
    await delay(100);
    res.send('Executed zoom in operation');
  } catch (err) {
    res.status(500).send(err.toString());
  }
});

app.get('/api/zoom_out', async (req, res) => {
  try {
    const viewport = page.viewport();
    const centerX = viewport.width / 2;
    const centerY = viewport.height / 2;
    await page.mouse.click(centerX, centerY);
    // Simulate mouse wheel scrolling down for zoom out
    await page.mouse.wheel({ deltaY: 100 });
    await delay(100);
    res.send('Executed zoom out operation');
  } catch (err) {
    res.status(500).send(err.toString());
  }
});

app.listen(API_PORT, () => {
  console.log(`API service running on port ${API_PORT}`);
});
