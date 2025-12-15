// slides-to-images.js
// Converts HTML presentation slides to individual PNG images

const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

async function convertSlidesToImages() {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    
    // Set viewport to presentation size
    await page.setViewport({
        width: 1920,
        height: 1080,
        deviceScaleFactor: 2 // Higher quality
    });
    
    // Look for HTML file in current directory
    const currentDir = __dirname;
    const possibleFiles = [
        'presentation.html',
        'index.html',
        'slides.html',
        'Heart Disease Predictor - Presentation.html'
    ];
    
    let htmlPath = null;
    
    // Find the HTML file
    for (const filename of possibleFiles) {
        const testPath = path.resolve(currentDir, filename);
        if (fs.existsSync(testPath)) {
            htmlPath = testPath;
            console.log(`Found HTML file: ${filename}`);
            break;
        }
    }
    
    // If not found, list available HTML files
    if (!htmlPath) {
        console.log('\n❌ Could not find presentation HTML file!');
        console.log('\nHTML files in current directory:');
        const files = fs.readdirSync(currentDir);
        const htmlFiles = files.filter(f => f.endsWith('.html'));
        
        if (htmlFiles.length === 0) {
            console.log('  No HTML files found!');
            console.log('\nPlease make sure your HTML file is in this directory:');
            console.log(`  ${currentDir}`);
        } else {
            htmlFiles.forEach(f => console.log(`  - ${f}`));
            console.log('\nUpdate the script with the correct filename or rename your file to "presentation.html"');
        }
        
        await browser.close();
        return;
    }
    
    // Load your HTML file
    const fileUrl = `file:///${htmlPath.replace(/\\/g, '/')}`;
    console.log(`Loading: ${fileUrl}`);
    
    await page.goto(fileUrl, { waitUntil: 'networkidle0' });
    
    // Create output directory
    const outputDir = path.resolve(__dirname, 'slides-output');
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir);
    }
    
    // Get total number of slides
    const totalSlides = await page.evaluate(() => {
        return document.querySelectorAll('.slide').length;
    });
    
    console.log(`Found ${totalSlides} slides. Converting...\n`);
    
    // Screenshot each slide
    for (let i = 0; i < totalSlides; i++) {
        console.log(`Processing slide ${i + 1}/${totalSlides}...`);
        
        // Navigate to slide
        await page.evaluate((slideIndex) => {
            const slides = document.querySelectorAll('.slide');
            slides.forEach(slide => slide.classList.remove('active'));
            slides[slideIndex].classList.add('active');
        }, i);
        
        // Wait a bit for animations
        await page.waitForTimeout(500);
        
        // Take screenshot of the slide container
        const slideElement = await page.$('.slideshow-container');
        await slideElement.screenshot({
            path: path.join(outputDir, `slide-${String(i + 1).padStart(2, '0')}.png`),
            type: 'png'
        });
        
        console.log(`✓ Saved slide ${i + 1}`);
    }
    
    await browser.close();
    console.log(`\n✅ Done! All ${totalSlides} slides saved to ${outputDir}`);
}

convertSlidesToImages().catch(console.error);