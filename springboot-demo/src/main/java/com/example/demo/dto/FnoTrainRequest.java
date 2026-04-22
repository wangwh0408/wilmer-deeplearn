package com.example.demo.dto;

import lombok.Data;

import java.io.Serializable;
import java.util.List;

@Data
public class FnoTrainRequest implements Serializable {

    private static final long serialVersionUID = 1L;

    private String framework = "pytorch";

    private String dataRoot;

    private List<String> problems;

    private List<String> categories;

    private Integer maxCases;

    private Integer modes1 = 12;

    private Integer modes2 = 12;

    private Integer width = 32;

    private Integer nLayers = 4;

    private Integer hiddenDim = 128;

    private Double learningRate = 0.001;

    private Double weightDecay = 0.0001;

    private Integer batchSize = 8;

    private Integer epochs = 100;

    private Integer schedulerStepSize = 50;

    private Double schedulerGamma = 0.5;

    private Integer inputSteps = 1;

    private Integer outputSteps = 1;

    private Double trainRatio = 0.8;

    private Boolean useCoordinates = true;

    private Boolean useMask = false;

    private Boolean normalize = true;

    private Integer seed = 42;

    private String modelSavePath;

    private String logFile;

    private Boolean verbose = true;
}
