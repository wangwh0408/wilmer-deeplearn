package com.example.demo.evaluation.entity;

import lombok.Data;

import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;

@Data
public class EvaluationTask implements Serializable {

    private static final long serialVersionUID = 1L;

    public enum Status {
        PENDING,
        RUNNING,
        COMPLETED,
        FAILED,
        CANCELLED
    }

    private String taskId;
    private Status status;
    private LocalDateTime startTime;
    private LocalDateTime endTime;
    private Long durationMillis;
    private String command;
    private Integer exitCode;
    private String errorMessage;

    private String modelPath;
    private String datasetPath;
    private Double samplePercentage;
    private List<String> problems;
    private List<String> categories;
    private Integer maxCases;
    private Boolean normalize;
    private String device;
    private String outputReportPath;

    private EvaluationMetrics metrics;
    private final List<EvaluationLogEntry> logs = new CopyOnWriteArrayList<>();
    private final transient AtomicInteger sequenceCounter = new AtomicInteger(0);

    private volatile int lastReadIndex = -1;

    @Data
    public static class EvaluationMetrics implements Serializable {
        private static final long serialVersionUID = 1L;
        private Integer totalSamples;
        private ComponentMetrics uComponent;
        private ComponentMetrics vComponent;
        private Map<String, Double> uMetrics;
        private Map<String, Double> vMetrics;
        private Map<String, Double> combinedMetrics;
        private Map<String, Double> uPerSampleStats;
        private Map<String, Double> vPerSampleStats;
        private Map<String, Object> bestWorstSamples;
        private String evaluationTime;
        private Double averageLoss;
        private String modelPath;
        private String dataRoot;
        private Integer numSamples;
        private Map<String, Object> config;
        private Map<String, Object> normalizationParams;

        @Data
        public static class ComponentMetrics implements Serializable {
            private static final long serialVersionUID = 1L;
            private String component;
            private Integer samples;
            private Double avgMseLoss;
            private MetricsDetail normalized;
            private MetricsDetail denormalized;
            private PerSampleStats perSampleStats;
            private BestWorstSample bestSample;
            private BestWorstSample worstSample;
            private Map<String, Double> dynamicMetrics;
        }

        @Data
        public static class MetricsDetail implements Serializable {
            private static final long serialVersionUID = 1L;
            private Double mse;
            private Double rmse;
            private Double mae;
            private Double r2;
            private Double mape;
            private Map<String, Double> dynamicMetrics;
        }

        @Data
        public static class PerSampleStats implements Serializable {
            private static final long serialVersionUID = 1L;
            private Double mseMean;
            private Double mseStd;
            private Double r2Mean;
            private Double r2Std;
            private Double maeMean;
            private Map<String, Double> dynamicStats;
        }

        @Data
        public static class BestWorstSample implements Serializable {
            private static final long serialVersionUID = 1L;
            private Integer index;
            private String caseName;
            private Integer timeStep;
            private Double mse;
            private Double r2;
            private Double rmse;
            private Double mae;
            private Double mape;
            private Map<String, Double> dynamicMetrics;
        }
    }

    public EvaluationTask() {
        this.status = Status.PENDING;
    }

    public EvaluationTask(String taskId) {
        this();
        this.taskId = taskId;
    }

    public int getNextSequence() {
        return sequenceCounter.getAndIncrement();
    }

    public void addLog(EvaluationLogEntry logEntry) {
        if (logEntry.getSequence() == null) {
            logEntry.setSequence(getNextSequence());
        }
        this.logs.add(logEntry);
    }

    public List<EvaluationLogEntry> getLogsOrderedBySequence() {
        List<EvaluationLogEntry> sortedLogs = new ArrayList<>(this.logs);
        Collections.sort(sortedLogs, new Comparator<EvaluationLogEntry>() {
            @Override
            public int compare(EvaluationLogEntry a, EvaluationLogEntry b) {
                if (a.getSequence() == null && b.getSequence() == null) {
                    if (a.getTimestamp() != null && b.getTimestamp() != null) {
                        return a.getTimestamp().compareTo(b.getTimestamp());
                    }
                    return 0;
                }
                if (a.getSequence() == null) return 1;
                if (b.getSequence() == null) return -1;
                return a.getSequence().compareTo(b.getSequence());
            }
        });
        return sortedLogs;
    }

    public List<EvaluationLogEntry> getLogs() {
        return getLogsOrderedBySequence();
    }

    public List<EvaluationLogEntry> getLogsSince(int fromIndex) {
        List<EvaluationLogEntry> orderedLogs = getLogsOrderedBySequence();
        if (fromIndex < 0 || fromIndex >= orderedLogs.size()) {
            return new ArrayList<>();
        }
        return new ArrayList<>(orderedLogs.subList(fromIndex, orderedLogs.size()));
    }

    public List<EvaluationLogEntry> getNewLogs() {
        List<EvaluationLogEntry> orderedLogs = getLogsOrderedBySequence();
        int currentIndex = orderedLogs.size();
        
        if (lastReadIndex >= 0 && lastReadIndex < currentIndex) {
            List<EvaluationLogEntry> newLogs = new ArrayList<>(
                orderedLogs.subList(lastReadIndex, currentIndex)
            );
            lastReadIndex = currentIndex;
            return newLogs;
        }
        
        lastReadIndex = currentIndex;
        return new ArrayList<>();
    }

    public int getLogCount() {
        return this.logs.size();
    }

    public int getMaxSequence() {
        int max = -1;
        for (EvaluationLogEntry entry : this.logs) {
            if (entry.getSequence() != null && entry.getSequence() > max) {
                max = entry.getSequence();
            }
        }
        return max;
    }

    public void markRunning() {
        this.status = Status.RUNNING;
        this.startTime = LocalDateTime.now();
    }

    public void markCompleted() {
        this.status = Status.COMPLETED;
        this.endTime = LocalDateTime.now();
        if (this.startTime != null) {
            this.durationMillis = java.time.Duration.between(this.startTime, this.endTime).toMillis();
        }
    }

    public void markFailed(String errorMessage) {
        this.status = Status.FAILED;
        this.endTime = LocalDateTime.now();
        this.errorMessage = errorMessage;
        if (this.startTime != null) {
            this.durationMillis = java.time.Duration.between(this.startTime, this.endTime).toMillis();
        }
    }

    public void markCancelled() {
        this.status = Status.CANCELLED;
        this.endTime = LocalDateTime.now();
        if (this.startTime != null) {
            this.durationMillis = java.time.Duration.between(this.startTime, this.endTime).toMillis();
        }
    }

    public boolean isRunning() {
        return status == Status.RUNNING;
    }

    public boolean isFinished() {
        return status == Status.COMPLETED || status == Status.FAILED || status == Status.CANCELLED;
    }
}
