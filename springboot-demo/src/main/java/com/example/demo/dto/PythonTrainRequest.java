package com.example.demo.dto;

import lombok.Data;

import java.io.Serializable;

@Data
public class PythonTrainRequest implements Serializable {

    private static final long serialVersionUID = 1L;

    private Integer modes = 12;
    private Integer width = 32;
    private Integer epochs = 50;
    private Integer batchSize = 16;
    private Double learningRate = 0.001;
    private Double weightDecay = 0.0001;
    private Integer resolution = 64;
    private Integer nTrainSamples = 800;
    private Integer nTestSamples = 200;
    private Integer schedulerStepSize = 20;
    private Double schedulerGamma = 0.5;
    private String modelSavePath = "fno2_model.pth";
    private String lossPlotPath = "training_loss.png";
    private Boolean verbose = true;
    private Boolean quickTrain = false;
}
