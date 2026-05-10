package com.example.demo.entity;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ModelServerEntity implements Serializable {

    private static final long serialVersionUID = 1L;

    private Long id;

    private String serverId;

    private Status status;

    private String modelPath;

    private String framework;

    private Integer port;

    private String host;

    private Integer pid;

    private String command;

    private String device;

    private ServerType serverType;

    private LocalDateTime startTime;

    private LocalDateTime stopTime;

    private String errorMessage;

    private LocalDateTime createTime;

    private LocalDateTime updateTime;

    public void onCreate() {
        createTime = LocalDateTime.now();
        updateTime = LocalDateTime.now();
    }

    public void onUpdate() {
        updateTime = LocalDateTime.now();
    }

    public enum Status {
        STARTING,
        RUNNING,
        STOPPING,
        STOPPED,
        FAILED
    }

    public enum ServerType {
        FLASK,
        FASTAPI
    }
}
