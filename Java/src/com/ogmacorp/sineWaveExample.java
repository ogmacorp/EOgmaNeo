// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

import java.io.File;
import java.lang.System;

import com.ogmacorp.eogmaneo.*;

public class sineWaveExample {

    static private boolean serializationEnabled = false;

    public static void main(String[] args) {

        // Create the main system interface, using 4 CPU cores
        ComputeSystem system = new ComputeSystem(4);

        int chunkSize = 8;

        int unitsPerChunk = chunkSize * chunkSize;

        // Range of input values
        double[] bounds = {-1.0, 1.0};

        System.out.println("Constructing the hierarchy...");
        StdVecLayerDesc lds = new StdVecLayerDesc();

        for (int i = 0; i < 3; i++) {
            LayerDesc ld = new LayerDesc();

            ld.set_width(16);
            ld.set_height(16);
            ld.set_chunkSize(8);
            ld.set_forwardRadius(12);
            ld.set_backwardRadius(12);
            ld.set_alpha(0.1f);
            ld.set_beta(0.01f);
            ld.set_temporalHorizon(2);

            lds.add(ld);
        }

        Hierarchy h = new Hierarchy();

        StdPairi inputSize = new StdPairi();
        inputSize.setFirst(chunkSize);
        inputSize.setSecond(chunkSize);

        StdVecPairi inputSizes = new StdVecPairi();
        inputSizes.add(inputSize);

        StdVeci inputChunkSizes = new StdVeci();
        inputChunkSizes.add(chunkSize);

        StdVecb predictInput = new StdVecb();
        predictInput.add(true);

        h.create(inputSizes, inputChunkSizes, predictInput, lds, 123);

        if (serializationEnabled) {
            File f = new File("sineSave.eohr");
            if (f.exists() && !f.isDirectory()) {
                System.out.println("Loading hierarchy from sineSave.eohr");
                h.load("sineSave.eohr");
            }
        } else {
            StdVeci chunkedSDR = new StdVeci();
            chunkedSDR.add(0);

            // Present the sine wave sequence for N steps
            System.out.println("Stepping the hierarchy...");
            for (int t = 0; t < 5000; t++) {
                double valueToEncode = Math.sin(t * 0.02 * 2.0 * Math.PI); //Test value

                // Single - chunk SDR
                chunkedSDR.set(0, (int) ((valueToEncode - bounds[0]) / (bounds[1] - bounds[0]) * (unitsPerChunk - 1) + 0.5));

                Std2DVeci inputValueList = new Std2DVeci();
                inputValueList.add(chunkedSDR);

                h.step(inputValueList, system, true);
            }
        }

        // Recall
        System.out.println("Predicting values...");
        for (int t = 0; t < 100; t++) {
            StdVeci predSDR = h.getPredictions(0); // First (only in this case) input layer prediction

            // Decode value
            double value = (double)predSDR.get(0) / (unitsPerChunk - 1) * (bounds[1] - bounds[0]) + bounds[0];

            System.out.printf(" %.2f", value);

            Std2DVeci inputValueList = new Std2DVeci();
            inputValueList.add(predSDR);
            h.step(inputValueList, system, false);
        }

        if (serializationEnabled) {
            System.out.println("Saving hierarchy to sineSave.eohr");
            h.save("sineSave.eohr");
        }

    }
}
