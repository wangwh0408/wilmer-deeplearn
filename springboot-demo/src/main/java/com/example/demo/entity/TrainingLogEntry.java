package com.example.demo.entity;

import lombok.Data;

import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;

@Data
public class TrainingLogEntry implements Serializable {

    private static final long serialVersionUID = 1L;

    private String taskId;
    private Integer sequence;
    private LocalDateTime timestamp;
    private String level;
    private String message;
    private String source;
    private String streamType;

    public TrainingLogEntry() {
        this.timestamp = LocalDateTime.now();
    }

    public TrainingLogEntry(String taskId, Integer sequence, String level, String message, String source) {
        this.taskId = taskId;
        this.sequence = sequence;
        this.timestamp = LocalDateTime.now();
        this.level = level;
        this.message = message;
        this.source = source;
    }

    public TrainingLogEntry(String taskId, Integer sequence, String level, String message, String source, String streamType) {
        this.taskId = taskId;
        this.sequence = sequence;
        this.timestamp = LocalDateTime.now();
        this.level = level;
        this.message = message;
        this.source = source;
        this.streamType = streamType;
    }

    public static TrainingLogEntry info(String taskId, Integer sequence, String message) {
        return new TrainingLogEntry(taskId, sequence, "INFO", message, "SYSTEM");
    }

    public static TrainingLogEntry error(String taskId, Integer sequence, String message) {
        return new TrainingLogEntry(taskId, sequence, "ERROR", message, "SYSTEM");
    }

    public static TrainingLogEntry warn(String taskId, Integer sequence, String message) {
        return new TrainingLogEntry(taskId, sequence, "WARN", message, "SYSTEM");
    }

    public static TrainingLogEntry pythonOutput(String taskId, Integer sequence, String message) {
        return new TrainingLogEntry(taskId, sequence, "INFO", message, "PYTHON", "STDOUT");
    }

    public static TrainingLogEntry pythonError(String taskId, Integer sequence, String message) {
        return new TrainingLogEntry(taskId, sequence, "ERROR", message, "PYTHON", "STDERR");
    }

    @Deprecated
    public static TrainingLogEntry info(String taskId, String message) {
        return info(taskId, null, message);
    }

    @Deprecated
    public static TrainingLogEntry error(String taskId, String message) {
        return error(taskId, null, message);
    }

    @Deprecated
    public static TrainingLogEntry warn(String taskId, String message) {
        return warn(taskId, null, message);
    }

    @Deprecated
    public static TrainingLogEntry pythonOutput(String taskId, String message) {
        return pythonOutput(taskId, null, message);
    }

    @Deprecated
    public static TrainingLogEntry pythonError(String taskId, String message) {
        return pythonError(taskId, null, message);
    }
}
