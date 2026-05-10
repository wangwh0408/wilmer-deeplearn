package com.example.demo.evaluation.entity;

import lombok.Data;

import java.io.Serializable;
import java.time.LocalDateTime;

@Data
public class EvaluationLogEntry implements Serializable {

    private static final long serialVersionUID = 1L;

    private String taskId;
    private Integer sequence;
    private LocalDateTime timestamp;
    private String level;
    private String message;
    private String source;
    private String streamType;

    public static EvaluationLogEntry info(String taskId, Integer sequence, String message) {
        EvaluationLogEntry entry = new EvaluationLogEntry();
        entry.setTaskId(taskId);
        entry.setSequence(sequence);
        entry.setTimestamp(LocalDateTime.now());
        entry.setLevel("INFO");
        entry.setMessage(message);
        entry.setSource("EVALUATION");
        entry.setStreamType("STDOUT");
        return entry;
    }

    public static EvaluationLogEntry error(String taskId, Integer sequence, String message) {
        EvaluationLogEntry entry = new EvaluationLogEntry();
        entry.setTaskId(taskId);
        entry.setSequence(sequence);
        entry.setTimestamp(LocalDateTime.now());
        entry.setLevel("ERROR");
        entry.setMessage(message);
        entry.setSource("EVALUATION");
        entry.setStreamType("STDERR");
        return entry;
    }

    public static EvaluationLogEntry warning(String taskId, Integer sequence, String message) {
        EvaluationLogEntry entry = new EvaluationLogEntry();
        entry.setTaskId(taskId);
        entry.setSequence(sequence);
        entry.setTimestamp(LocalDateTime.now());
        entry.setLevel("WARN");
        entry.setMessage(message);
        entry.setSource("EVALUATION");
        entry.setStreamType("STDOUT");
        return entry;
    }

    public static EvaluationLogEntry pythonOutput(String taskId, Integer sequence, String message) {
        EvaluationLogEntry entry = new EvaluationLogEntry();
        entry.setTaskId(taskId);
        entry.setSequence(sequence);
        entry.setTimestamp(LocalDateTime.now());
        entry.setLevel("INFO");
        entry.setMessage(message);
        entry.setSource("PYTHON");
        entry.setStreamType("STDOUT");
        return entry;
    }
}
