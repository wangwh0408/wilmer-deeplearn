package com.example.demo.dto;

import lombok.Data;

import java.io.Serializable;

@Data
public class Fno2EnhancedTrainRequest implements Serializable {

    private static final long serialVersionUID = 1L;

    private String dataRoot = "C:\\traework\\data";

    private String modelOutput = "fno2_full_model.pth";

    private String logFile = "training_full.log";

    private String problems = "cavity";

    private String categories;

    private Integer classNum = 3;

    private Integer epochs = 30;

    private Integer batchSize = 8;

    private Double learningRate = 0.001;

    private Integer modes = 12;

    private Integer width = 32;

    private Integer maxCasesPerCategory = 10;

    private Double trainRatio = 0.8;

    private Integer printEvery = 1;

    private String device;

    private Boolean noGradientStats = false;

    private Boolean noParamStats = false;

    private Boolean noTensorStats = false;

    private Boolean noPredictionSamples = false;

    private String logLevel = "INFO";

    private Boolean quiet = false;

    private Boolean verbose = false;

    private Boolean noFileLog = false;
}
