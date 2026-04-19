package com.example.demo.entity;

import lombok.Data;

import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

@Data
public class PythonTrainResult implements Serializable {

    private static final long serialVersionUID = 1L;

    private boolean success;
    private Integer exitCode;
    private String output;
    private String errorOutput;
    private LocalDateTime startTime;
    private LocalDateTime endTime;
    private Long durationMillis;
    private String command;
    private Map<String, Object> parsedResult;
    private String modelPath;
    private Double finalTrainLoss;
    private Double finalTestLoss;
    private List<String> trainLosses;
    private List<String> testLosses;
    private String errorMessage;
}
