package com.example.demo.entity;

import lombok.Data;

import javax.persistence.*;
import java.io.Serializable;
import java.time.LocalDateTime;

@Data
@Entity
@Table(name = "t_training_task")
public class TrainingTaskEntity implements Serializable {

    private static final long serialVersionUID = 1L;

    public enum Status {
        PENDING,
        RUNNING,
        COMPLETED,
        FAILED,
        CANCELLED
    }

    public enum TaskType {
        TRAINING,
        EVALUATION,
        PADDLE_TRAINING
    }

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "task_id", unique = true, nullable = false, length = 50)
    private String taskId;

    @Enumerated(EnumType.STRING)
    @Column(name = "status", length = 20)
    private Status status;

    @Enumerated(EnumType.STRING)
    @Column(name = "task_type", length = 20)
    private TaskType taskType;

    @Column(name = "framework", length = 50)
    private String framework;

    @Column(name = "command", columnDefinition = "TEXT")
    private String command;

    @Column(name = "start_time")
    private LocalDateTime startTime;

    @Column(name = "end_time")
    private LocalDateTime endTime;

    @Column(name = "duration_millis")
    private Long durationMillis;

    @Column(name = "exit_code")
    private Integer exitCode;

    @Column(name = "error_message", columnDefinition = "TEXT")
    private String errorMessage;

    @Column(name = "final_train_loss")
    private Double finalTrainLoss;

    @Column(name = "final_test_loss")
    private Double finalTestLoss;

    @Column(name = "best_train_loss")
    private Double bestTrainLoss;

    @Column(name = "best_test_loss")
    private Double bestTestLoss;

    @Column(name = "model_path", length = 500)
    private String modelPath;

    @Column(name = "log_count")
    private Integer logCount = 0;

    @Column(name = "create_time")
    private LocalDateTime createTime;

    @Column(name = "update_time")
    private LocalDateTime updateTime;

    @PrePersist
    protected void onCreate() {
        createTime = LocalDateTime.now();
        updateTime = LocalDateTime.now();
        if (status == null) {
            status = Status.PENDING;
        }
    }

    @PreUpdate
    protected void onUpdate() {
        updateTime = LocalDateTime.now();
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

    public void incrementLogCount() {
        if (this.logCount == null) {
            this.logCount = 0;
        }
        this.logCount++;
    }
}
