package com.example.demo.evaluation.dto;

import lombok.Data;

import java.util.List;

@Data
public class EvaluationRequest {

    private String modelPath;
    private String datasetPath;
    private Double samplePercentage;
    private List<String> problems;
    private List<String> categories;
    private Integer maxCases;
    private Integer maxCasesPerCategory;
    private Boolean normalize;
    private String device;
    private String outputReportPath;
    private String workingDirectory;
    private List<String> metrics;
    private String logLevel;
    private Boolean noFileLog;
    private Integer printEvery;
    private Integer batchSize;
}
