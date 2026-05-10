package com.example.demo.dto;

import lombok.Data;

import java.io.Serializable;

@Data
public class Fno2EvaluateRequest implements Serializable {

    private static final long serialVersionUID = 1L;

    private String modelPath = "fno2_full_model.pth";

    private String dataRoot = "C:\\traework\\data";

    private String logFile = "evaluation_enhanced.log";

    private String outputReport = "evaluation_report_enhanced.json";

    private String problems = "cavity";

    private String categories = "bc,geo,prop";

    private Integer maxCasesPerCategory;

    private Integer batchSize = 8;

    private Integer printEvery = 10;

    private String device;

    private String metrics = "mse,rmse,mae,r2,mape";

    private String logLevel = "INFO";

    private Boolean quiet = false;

    private Boolean verbose = false;

    private Boolean noFileLog = false;

    private Boolean listMetrics = false;
}
