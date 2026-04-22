package com.example.demo.entity;

import lombok.Data;

import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantReadWriteLock;

@Data
public class TrainingTask implements Serializable {

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
    private String framework;
    private String command;
    private Integer exitCode;
    private String errorMessage;

    private Double finalTrainLoss;
    private Double finalTestLoss;
    private Double bestTrainLoss;
    private Double bestTestLoss;
    private String modelPath;

    private final List<TrainingLogEntry> logs = new CopyOnWriteArrayList<>();
    private transient final AtomicInteger sequenceCounter = new AtomicInteger(0);
    private transient final ReentrantReadWriteLock logLock = new ReentrantReadWriteLock();

    private volatile int lastReadIndex = -1;

    public TrainingTask() {
        this.status = Status.PENDING;
    }

    public TrainingTask(String taskId) {
        this();
        this.taskId = taskId;
    }

    public int getNextSequence() {
        return sequenceCounter.getAndIncrement();
    }

    public void addLog(TrainingLogEntry logEntry) {
        if (logEntry.getSequence() == null) {
            logEntry.setSequence(getNextSequence());
        }
        this.logs.add(logEntry);
    }

    public List<TrainingLogEntry> getLogsOrderedBySequence() {
        List<TrainingLogEntry> sortedLogs = new ArrayList<>(this.logs);
        Collections.sort(sortedLogs, new Comparator<TrainingLogEntry>() {
            @Override
            public int compare(TrainingLogEntry a, TrainingLogEntry b) {
                if (a.getSequence() == null && b.getSequence() == null) {
                    if (a.getTimestamp() != null && b.getTimestamp() != null) {
                        return a.getTimestamp().compareTo(b.getTimestamp());
                    }
                    return 0;
                }
                if (a.getSequence() == null) {
                    return 1;
                }
                if (b.getSequence() == null) {
                    return -1;
                }
                return a.getSequence().compareTo(b.getSequence());
            }
        });
        return sortedLogs;
    }

    public List<TrainingLogEntry> getLogs() {
        return getLogsOrderedBySequence();
    }

    public List<TrainingLogEntry> getLogsSince(int fromIndex) {
        List<TrainingLogEntry> orderedLogs = getLogsOrderedBySequence();
        if (fromIndex < 0 || fromIndex >= orderedLogs.size()) {
            return new ArrayList<>();
        }
        return new ArrayList<>(orderedLogs.subList(fromIndex, orderedLogs.size()));
    }

    public List<TrainingLogEntry> getLogsSinceSequence(int fromSequence) {
        List<TrainingLogEntry> result = new ArrayList<>();
        List<TrainingLogEntry> orderedLogs = getLogsOrderedBySequence();
        
        for (TrainingLogEntry entry : orderedLogs) {
            if (entry.getSequence() != null && entry.getSequence() > fromSequence) {
                result.add(entry);
            }
        }
        return result;
    }

    public List<TrainingLogEntry> getNewLogs() {
        List<TrainingLogEntry> orderedLogs = getLogsOrderedBySequence();
        int currentIndex = orderedLogs.size();
        
        if (lastReadIndex >= 0 && lastReadIndex < currentIndex) {
            List<TrainingLogEntry> newLogs = new ArrayList<>(
                orderedLogs.subList(lastReadIndex, currentIndex)
            );
            lastReadIndex = currentIndex;
            return newLogs;
        }
        
        lastReadIndex = currentIndex;
        return new ArrayList<>();
    }

    public List<TrainingLogEntry> getNewLogsBySequence() {
        List<TrainingLogEntry> allLogs = getLogsOrderedBySequence();
        if (allLogs.isEmpty()) {
            return new ArrayList<>();
        }
        
        int maxSequence = -1;
        for (TrainingLogEntry entry : allLogs) {
            if (entry.getSequence() != null && entry.getSequence() > maxSequence) {
                maxSequence = entry.getSequence();
            }
        }
        
        int currentMaxSequence = maxSequence;
        if (currentMaxSequence <= lastReadIndex) {
            return new ArrayList<>();
        }
        
        List<TrainingLogEntry> result = new ArrayList<>();
        for (TrainingLogEntry entry : allLogs) {
            if (entry.getSequence() != null && entry.getSequence() > lastReadIndex) {
                result.add(entry);
            }
        }
        
        lastReadIndex = currentMaxSequence;
        return result;
    }

    public void resetLastReadIndex() {
        this.lastReadIndex = -1;
    }

    public int getLogCount() {
        return this.logs.size();
    }

    public int getMaxSequence() {
        int max = -1;
        for (TrainingLogEntry entry : this.logs) {
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
