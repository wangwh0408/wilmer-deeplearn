package com.example.demo.entity;

import lombok.Data;

import javax.persistence.*;
import java.io.Serializable;
import java.time.LocalDateTime;

@Data
@Entity
@Table(name = "t_training_log", indexes = {
    @Index(name = "idx_task_id", columnList = "task_id"),
    @Index(name = "idx_task_sequence", columnList = "task_id, sequence"),
    @Index(name = "idx_created_at", columnList = "created_at")
})
public class TrainingLogEntryEntity implements Serializable {

    private static final long serialVersionUID = 1L;

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "task_id", nullable = false, length = 50)
    private String taskId;

    @Column(name = "sequence")
    private Integer sequence;

    @Column(name = "log_level", length = 20)
    private String level;

    @Column(name = "message", columnDefinition = "TEXT")
    private String message;

    @Column(name = "source", length = 20)
    private String source;

    @Column(name = "stream_type", length = 20)
    private String streamType;

    @Column(name = "created_at")
    private LocalDateTime createdAt;

    @PrePersist
    protected void onCreate() {
        if (createdAt == null) {
            createdAt = LocalDateTime.now();
        }
    }

    public static TrainingLogEntryEntity fromTrainingLogEntry(String taskId, TrainingLogEntry entry) {
        TrainingLogEntryEntity entity = new TrainingLogEntryEntity();
        entity.setTaskId(taskId);
        entity.setSequence(entry.getSequence());
        entity.setLevel(entry.getLevel());
        entity.setMessage(entry.getMessage());
        entity.setSource(entry.getSource());
        entity.setStreamType(entry.getStreamType());
        entity.setCreatedAt(entry.getTimestamp());
        return entity;
    }

    public static TrainingLogEntryEntity createPythonOutput(String taskId, Integer sequence, String message) {
        TrainingLogEntryEntity entity = new TrainingLogEntryEntity();
        entity.setTaskId(taskId);
        entity.setSequence(sequence);
        entity.setLevel("INFO");
        entity.setMessage(message);
        entity.setSource("PYTHON");
        entity.setStreamType("STDOUT");
        return entity;
    }

    public static TrainingLogEntryEntity createPythonError(String taskId, Integer sequence, String message) {
        TrainingLogEntryEntity entity = new TrainingLogEntryEntity();
        entity.setTaskId(taskId);
        entity.setSequence(sequence);
        entity.setLevel("ERROR");
        entity.setMessage(message);
        entity.setSource("PYTHON");
        entity.setStreamType("STDERR");
        return entity;
    }

    public static TrainingLogEntryEntity createSystemInfo(String taskId, Integer sequence, String message) {
        TrainingLogEntryEntity entity = new TrainingLogEntryEntity();
        entity.setTaskId(taskId);
        entity.setSequence(sequence);
        entity.setLevel("INFO");
        entity.setMessage(message);
        entity.setSource("SYSTEM");
        return entity;
    }

    public static TrainingLogEntryEntity createSystemError(String taskId, Integer sequence, String message) {
        TrainingLogEntryEntity entity = new TrainingLogEntryEntity();
        entity.setTaskId(taskId);
        entity.setSequence(sequence);
        entity.setLevel("ERROR");
        entity.setMessage(message);
        entity.setSource("SYSTEM");
        return entity;
    }
}
