// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <time.h>
#include <iostream>
#include <random>
#include <thread>

#include <Hierarchy.h>
#include <ImageEncoder.h>
#include <VisAdapter.h>
using namespace eogmaneo;


// Restrict input image dimensions
const int imageWidth = 128;
const int imageHeight = 128;

// Display status text plus two columns and multiple rows
const unsigned int windowWidth = (imageWidth + 2) * 3.5;
const unsigned int windowHeight = 72 + (imageHeight + 2);

// X Offset for all window drawing
const float xOffset = 96.0f;
const int progressBarLength = 40;

// Predictive hierarchy settings
const int numLayers = 4;
const int hiddenWidth = 32;
const int hiddenHeight = 32;
const int chunkSize = 4;
const int radius = 9;

static std::shared_ptr<ComputeSystem> cs;
static ImageEncoder imagePreEncoder;
static Hierarchy h;

// Optional connection to NeoVis application
const bool connectToNeoVis = false;
static VisAdapter *visAdapter = NULL;


void ConstructEOgmaNeoHierarchy() {

    std::cout << "Constructing EOgmaNeo hierarchy and pre-encoder" << std::endl;

    unsigned int nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0) nthreads = 1;

    std::cout << "Using " << nthreads << " workers" << std::endl;
    std::cout << std::endl;

    cs = std::make_shared<ComputeSystem>(nthreads);

    imagePreEncoder.create(imageWidth, imageHeight, hiddenWidth, hiddenHeight, chunkSize, 16, 123);

    std::vector<LayerDesc> lds;
    std::vector<bool> predictInputs;

    for (int l = 0; l < numLayers; l++)
    {
        LayerDesc ld;
        ld._width = hiddenWidth;
        ld._height = hiddenHeight;
        ld._chunkSize = chunkSize;

        ld._forwardRadius = radius;
        ld._backwardRadius = radius;

        ld._ticksPerUpdate = 2;
        ld._temporalHorizon = 2;

        ld._alpha = 0.4f;
        ld._beta = 0.4f;

        lds.push_back(ld);

        predictInputs.push_back(true);
    }

    std::vector<std::pair<int, int>> inputSizes;
    inputSizes.push_back(std::pair<int, int>{hiddenWidth, hiddenHeight});

    std::vector<int> inputChunkSizes;
    inputChunkSizes.push_back(chunkSize);

    h.create(inputSizes, inputChunkSizes, predictInputs, lds, 123);

    if (connectToNeoVis) {
        visAdapter = new VisAdapter();
        visAdapter->create(&h, 54000);
    }

}


int main() {

    // Create the SFML window
    sf::RenderWindow window;

    window.create(sf::VideoMode(windowWidth, windowHeight), "Video Prediction Test", sf::Style::Default);
    window.setVerticalSyncEnabled(true);
    window.setFramerateLimit(0);
    //window.setPosition(sf::Vector2i(64, 64));

    sf::Font font;

#if defined(_WINDOWS)
    font.loadFromFile("C:/Windows/Fonts/Arial.ttf");
#elif defined(__APPLE__)
    font.loadFromFile("/Library/Fonts/Courier New.ttf");
#else
    font.loadFromFile("/usr/share/fonts/truetype/freefont/FreeMono.ttf");
#endif

    std::string hierarchyFileName("Hierarchy.eohr");
    //std::string videoFileName("Clock-OneArm.mp4");
    std::string videoFileName("Tesseract.mp4");

    cv::VideoCapture capture(videoFileName);
    cv::Mat frame;

    if (!capture.isOpened()) {
        std::cerr << "Could not open capture: " << videoFileName << std::endl;
        return 1;
    }

    const int videoWidth = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    const int videoHeight = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));

    if (videoWidth != videoHeight) {
        std::cerr << "Video file " << videoFileName << " has a non-square frame!" << std::endl;
        return 1;
    }

    std::cout <<
        "This example shows how EOgmaNeo and the Image Pre-encoder can be used" << std::endl <<
        "to predict the next image from a video." << std::endl <<
        std::endl <<
        "Step 1: Show the image pre-encoder every frame of the video multiple times." << std::endl <<
        "  The Escape key can skip this pre-encoder training, once it looks like it is" << std::endl <<
        "  able to reconstruct enough frames of the video (comparing source video images" << std::endl <<
        "  on the left with reconstruction on the right)" << std::endl <<
        std::endl <<
        "Step 2: Uses the trained image pre-encoder to produce sparse chunked" << std::endl <<
        "  representation to send into the EOgmaNeo predictive hierarchy, and" << std::endl <<
        "  step the hierarchy with training enabled. Left image shows the source video" << std::endl <<
        "  frame, the right image shows the predicted video frame" << std::endl <<
        "  The Escape key can terminate this EOgmaNeo training step." << std::endl <<
        std::endl <<
        "Step 3: Uses only predictions from the EOgmaNeo predictive hierarchy as input" << std::endl <<
        "  to the hierarchy to predict the next video frame." << std::endl <<
        std::endl <<
        "  Note: The first two plays of the video during step 2 are slowed down!" << std::endl <<
        std::endl;

    // Video rescaling render target
    sf::RenderTexture rescaleRT;
    rescaleRT.create(imageWidth, imageHeight);

    // Create the PreEncoder and Hierarchy
    ConstructEOgmaNeoHierarchy();

    int captureLength = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));

    // Calculate actual number of frames
    int i = 1;
    for (; i <= captureLength; i++) {
        capture >> frame;

        if (frame.empty())
            break;
    }

    captureLength = i - 1;

    std::cout << "Video has " << captureLength << " frames" << std::endl << std::endl;

    int numPreEncoderIters = captureLength;
    int numIter = 16;// captureLength;

    // Increase number of presentations for the one-arm clock video
    if (videoFileName == "Clock-OneArm.mp4") {
        numPreEncoderIters *= 4;
    }
    else
    if (videoFileName == "Tesseract.mp4") {
        numPreEncoderIters /= 4;
    }


    //-------------------------------------
    // Step 1: Train the image pre-encoder

    bool quit = false;

    std::cout << "Pre-training image pre-encoder..." << std::endl;

    for (int j = 0; j < numPreEncoderIters; j++) {

        // Seek back to the start of the video
        capture.set(cv::CAP_PROP_POS_FRAMES, 0.0);

        for (i = 0; i < captureLength; i++) {

            sf::Event windowEvent;
            while (window.pollEvent(windowEvent)) {
                switch (windowEvent.type) {
                case sf::Event::Closed:
                    quit = true;
                    break;

                case sf::Event::KeyReleased:
                    if (windowEvent.key.code == sf::Keyboard::Escape) {
                        // Skip further training of the Image pre-encoder
                        j = numPreEncoderIters;
                        i = captureLength;
                    }
                    break;

                default:
                    break;
                }
            }

            window.clear(sf::Color::Black);

            std::string st;
            st += "Step: " + std::to_string(j + 1) + "/" + std::to_string(numPreEncoderIters) +
                "  Mode: Pre-encoder training";

            sf::Text t;
            t.setFont(font);
            t.setCharacterSize(16);
            t.setString(st);
            t.setPosition(12.0f, 12.0f);
            t.setFillColor(sf::Color::White);
            window.draw(t);

            capture >> frame;

            sf::Image img;
            img.create(frame.cols, frame.rows);

            for (unsigned int x = 0; x < img.getSize().x; x++)
                for (unsigned int y = 0; y < img.getSize().y; y++) {
                    sf::Uint8 r = frame.data[(x + y * img.getSize().x) * 3 + 2];
                    sf::Uint8 g = frame.data[(x + y * img.getSize().x) * 3 + 1];
                    sf::Uint8 b = frame.data[(x + y * img.getSize().x) * 3 + 0];
                    //img.setPixel(x, y, sf::Color(r, g, b));

                    sf::Uint8 v = static_cast<sf::Uint8>((0.2126f * (r / 256.0f) + 0.7152f * (g / 256.0f) + 0.0722f * (b / 256.0f)) * 256.0f);
                    img.setPixel(x, y, sf::Color(v, v, v));
                }

            // Convert to a SFML texture
            sf::Texture tex;
            tex.loadFromImage(img);
            tex.setSmooth(true);

            // Rescale captured frame into a smaller texture
            float scale = std::min(static_cast<float>(rescaleRT.getSize().x) / img.getSize().x, static_cast<float>(rescaleRT.getSize().y) / img.getSize().y);
            sf::Sprite s;
            s.setPosition(rescaleRT.getSize().x * 0.5f, rescaleRT.getSize().y * 0.5f);
            s.setTexture(tex);
            s.setOrigin(sf::Vector2f(tex.getSize().x * 0.5f, tex.getSize().y * 0.5f));
            s.setScale(scale, scale);

            rescaleRT.clear();
            rescaleRT.draw(s);
            rescaleRT.display();

            // Grab SFML image from rescaled texture
            sf::Image reImg = rescaleRT.getTexture().copyToImage();

            // Construct input buffer for the Image PreEncoder
            std::vector<float> input;
            for (unsigned int x = 0; x < reImg.getSize().x; x++)
                for (unsigned int y = 0; y < reImg.getSize().y; y++) {
                    sf::Color c = reImg.getPixel(x, y);
                    float greyScale = 0.2126f * (c.r / 256.0f) + 0.7152f * (c.g / 256.0f) + 0.0722f * (c.b / 256.0f);
                    input.push_back(greyScale);
                }

            std::vector<std::vector<int>> inputs;
            inputs.push_back(imagePreEncoder.activate(input, *cs));

            std::vector<float> output = imagePreEncoder.reconstruct(inputs[0], *cs);
            imagePreEncoder.learn(0.95f, *cs);

            st = "Source";
            t.setString(st);
            t.setPosition(xOffset + 12.0f, 48.0f);
            window.draw(t);

            st = "Reconstruction";
            t.setString(st);
            t.setPosition(xOffset + imageWidth + 12.0f, 48.0f);
            window.draw(t);

            // Draw the original (recaled) image frame
            sf::Sprite s1;
            s1.setPosition(xOffset + 0.f, 72.f);
            s1.setTexture(rescaleRT.getTexture());
            window.draw(s1);

            // Transfer the output array into a SFML image
            sf::Image reconstruct;
            reconstruct.create(imageWidth, imageHeight);
            for (unsigned int x = 0; x < reconstruct.getSize().x; x++)
                for (unsigned int y = 0; y < reconstruct.getSize().y; y++) {
                    sf::Uint8 v = static_cast<sf::Uint8>(output[(y + x * reconstruct.getSize().y)] * 256.f);
                    reconstruct.setPixel(x, y, sf::Color(v, v, v));
                }

            // Display the output image
            sf::Texture tex2;
            tex2.loadFromImage(reconstruct);
            sf::Sprite s2;
            s2.setPosition(xOffset + imageWidth + 2.f, 72.f);
            s2.setTexture(tex2);
            window.draw(s2);

            window.display();
        }

    }


    //-------------------------------------------
    // Step 2: Stepping the predictive hierarchy

    int currentFrame = 0;

    std::cout << "Iterating the predictive hierarchy..." << std::endl;

    for (int iter = 0; iter < numIter && !quit; iter++) {
        std::cout << "Iteration " << (iter + 1) << " of " << numIter << ":" << std::endl;

        capture.set(cv::CAP_PROP_POS_FRAMES, 0.0);
        currentFrame = 0;

        // Run through video
        do {
            sf::Event windowEvent;
            while (window.pollEvent(windowEvent)) {
                switch (windowEvent.type) {
                case sf::Event::Closed:
                    quit = true;
                    break;

                case sf::Event::KeyReleased:
                    if (windowEvent.key.code == sf::Keyboard::Escape)
                        quit = true;

                    if (windowEvent.key.code == sf::Keyboard::S)
                        h.save(hierarchyFileName);
                    break;

                default:
                    break;
                }
            }

            window.clear(sf::Color::Black);

            capture >> frame;
            currentFrame++;

            if (frame.empty())
                break;

            if (currentFrame > captureLength)
                break;

            // Show progress bar
            float ratio = static_cast<float>(currentFrame) / (captureLength + 1);

            // Console
            std::cout << "\r";
            std::cout << "[";

            int bars = static_cast<int>(std::round(ratio * progressBarLength));
            int spaces = progressBarLength - bars;

            for (int i = 0; i < bars; i++)
                std::cout << "=";

            for (int i = 0; i < spaces; i++)
                std::cout << " ";

            std::cout << "] " << static_cast<int>(ratio * 100.0f) << "%";

            // UI
            sf::RectangleShape rs;

            sf::Text t;
            t.setFont(font);
            t.setCharacterSize(16);
            t.setFillColor(sf::Color::White);

            // Show progress bar
            rs.setPosition(8.0f, 8.0f);
            rs.setSize(sf::Vector2f(128.0f * ratio, 32.0f));
            rs.setFillColor(sf::Color::Red);
            window.draw(rs);

            // Progress bar outline
            rs.setPosition(8.0f, 8.0f);
            rs.setFillColor(sf::Color::Transparent);
            rs.setOutlineColor(sf::Color::White);
            rs.setOutlineThickness(2.0f);
            rs.setSize(sf::Vector2f(imageWidth, 32.0f));
            window.draw(rs);

            // Status string
            std::string st;
            st += std::to_string(static_cast<int>(ratio * 100.0f)) +
                "% Pass: " + std::to_string(iter + 1) + " of " + std::to_string(numIter) +
                //" " + std::to_string(currentFrame) + "/" + std::to_string(captureLength) +
                "  Mode: predictions";

            t.setString(st);
            t.setPosition(144.0f, 12.0f);
            window.draw(t);


            // Convert video frame into SFML image
            sf::Image img;
            img.create(frame.cols, frame.rows);

            for (unsigned int x = 0; x < img.getSize().x; x++)
                for (unsigned int y = 0; y < img.getSize().y; y++) {
                    sf::Uint8 r = frame.data[(x + y * img.getSize().x) * 3 + 2];
                    sf::Uint8 g = frame.data[(x + y * img.getSize().x) * 3 + 1];
                    sf::Uint8 b = frame.data[(x + y * img.getSize().x) * 3 + 0];
                    //img.setPixel(x, y, sf::Color(r, g, b));

                    sf::Uint8 v = static_cast<sf::Uint8>((0.2126f * (r / 256.0f) + 0.7152f * (g / 256.0f) + 0.0722f * (b / 256.0f)) * 256.0f);
                    img.setPixel(x, y, sf::Color(v, v, v));
                }

            // Convert to a SFML texture
            sf::Texture tex;
            tex.loadFromImage(img);
            tex.setSmooth(true);

            // Rescale captured frame into a smaller texture
            float scale = std::min(static_cast<float>(rescaleRT.getSize().x) / img.getSize().x, static_cast<float>(rescaleRT.getSize().y) / img.getSize().y);
            sf::Sprite s;
            s.setPosition(rescaleRT.getSize().x * 0.5f, rescaleRT.getSize().y * 0.5f);
            s.setTexture(tex);
            s.setOrigin(sf::Vector2f(tex.getSize().x * 0.5f, tex.getSize().y * 0.5f));
            s.setScale(scale, scale);

            rescaleRT.clear();
            rescaleRT.draw(s);
            rescaleRT.display();

            // Get the SFML image from rescaled texture for the current video image (t+0)
            sf::Image reImg = rescaleRT.getTexture().copyToImage();


            // Construct an input buffer for the Image PreEncoder output for the current video frame
            std::vector<float> input;
            for (unsigned int x = 0; x < reImg.getSize().x; x++)
                for (unsigned int y = 0; y < reImg.getSize().y; y++) {
                    sf::Color c = reImg.getPixel(x, y);
                    float v = 0.2126f * (c.r / 256.0f) + 0.7152f * (c.g / 256.0f) + 0.0722f * (c.b / 256.0f);
                    input.push_back(v);
                }

            // Run the current (t+0) video frame through the pre-encoder
            std::vector<std::vector<int>> inputs;
            inputs.push_back(imagePreEncoder.activate(input, *cs));

            // Step the predictive hierarchy
            h.step(inputs, *cs, true);

            // Grab the prediction (t+1) sparse chunked representation
            std::vector<int> pred;
            pred = h.getPredictions(0);

            // Decode the prediction to an image using the pre-encoder
            std::vector<float> output;
            output = imagePreEncoder.reconstruct(pred, *cs);


            // Left images
            st = "Source";
            t.setString(st);
            t.setPosition(xOffset + 12.0f, 48.0f);
            window.draw(t);

            // Draw the original (recaled) image frame
            sf::Sprite s1;
            s1.setPosition(xOffset + 0.f, 72.f);
            s1.setTexture(rescaleRT.getTexture());
            window.draw(s1);

            st = "t+0";
            t.setString(st);
            t.setPosition(xOffset - 36.0f, (64.0f + imageHeight / 2));
            window.draw(t);

            st = "Prediction";
            t.setString(st);
            t.setPosition(xOffset + imageWidth + 12.0f, 48.0f);
            window.draw(t);

            // Transfer the output array into a SFML image
            sf::Image reconstruct;
            reconstruct.create(imageWidth, imageHeight);
            for (unsigned int x = 0; x < reconstruct.getSize().x; x++)
                for (unsigned int y = 0; y < reconstruct.getSize().y; y++) {
                    sf::Uint8 v = static_cast<sf::Uint8>(output[(y + x * reconstruct.getSize().y)] * 256.f);
                    reconstruct.setPixel(x, y, sf::Color(v, v, v));
                }

            // Display the predicted output image
            sf::Texture tex2;
            tex2.loadFromImage(reconstruct);
            sf::Sprite s2;
            s2.setPosition(xOffset + imageWidth + 2.f, 72.f);
            s2.setTexture(tex2);
            window.draw(s2);

            // Display the predicted output time
            st = "t+1";
            t.setString(st);
            t.setPosition(xOffset + (imageWidth * 2) + 12.0f, (64.0f + imageHeight / 2));
            window.draw(t);

            if (connectToNeoVis && visAdapter != NULL) {
                visAdapter->update(0.01f);
            }

            window.display();

            if (0) { //iter == 0 || iter == 1) {
                unsigned long msDelay = 500;
                sf::sleep(sf::milliseconds(msDelay));
            }
        } while (!frame.empty() && !quit);

        // Make sure bar is at 100%
        std::cout << "\r" << "[";

        for (int i = 0; i < progressBarLength; i++)
            std::cout << "=";

        std::cout << "] 100%" << std::endl;
    }


    //-----------------------------------------------------------
    // Step 3: Only send t+1 predictions back into the hierarchy

    quit = false;
    currentFrame = 0;

    std::vector<int> pred;
    pred = h.getPredictions(0);

    do {
        sf::Event windowEvent;

        while (window.pollEvent(windowEvent)) {
            switch (windowEvent.type) {
            case sf::Event::Closed:
                quit = true;
                break;

            case sf::Event::KeyReleased:
                if (windowEvent.key.code == sf::Keyboard::Escape)
                    quit = true;
                break;

            default:
                break;
            }
        }

        currentFrame++;

        window.clear();

        std::vector<std::vector<int>> inputs;
        inputs.push_back(h.getPredictions(0));

        h.step(inputs, *cs, false);

        pred = h.getPredictions(0);

        // Use the prediction to reconstruct the next image (output array)
        std::vector<float> output;
        output = imagePreEncoder.reconstruct(pred, *cs);

        // Transfer the output array into a SFML image
        sf::Image reconstruct;
        reconstruct.create(imageWidth, imageHeight);
        for (unsigned int x = 0; x < reconstruct.getSize().x; x++)
            for (unsigned int y = 0; y < reconstruct.getSize().y; y++) {
                sf::Uint8 v = static_cast<sf::Uint8>(output[(x + y * reconstruct.getSize().x)] * 256.f);
                reconstruct.setPixel(y, x, sf::Color(v, v, v));
            }

        // Display the output image
        sf::Texture tex2;
        tex2.loadFromImage(reconstruct);
        sf::Sprite s2;
        s2.setPosition(xOffset + imageWidth + 2.f, 72.f);
        s2.setTexture(tex2);
        window.draw(s2);

        std::string st;
        st += "Frame: " + std::to_string(currentFrame) + "  Mode: prediction only";

        sf::Text t;
        t.setFont(font);
        t.setCharacterSize(16);
        t.setString(st);
        t.setPosition(12.0f, 12.0f);
        t.setFillColor(sf::Color::White);
        window.draw(t);

        st = "Prediction";
        t.setString(st);
        t.setPosition(xOffset + imageWidth + 12.0f, 48.0f);
        window.draw(t);

        window.display();

        if (connectToNeoVis && visAdapter != NULL) {
            visAdapter->update(0.01f);
        }

    } while (!quit);

    return 0;
}
