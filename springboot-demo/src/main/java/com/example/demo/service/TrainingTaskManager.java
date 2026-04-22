package com.example.demo.service;

import com.example.demo.entity.TrainingTask;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

@Component
public class TrainingTaskManager {

    private final Map<String, TrainingTask> tasks = new ConcurrentHashMap<>();

    public String createTask() {
        String taskId = UUID.randomUUID().toString();
        TrainingTask task = new TrainingTask(taskId);
        tasks.put(taskId, task);
        return taskId;
    }

    public String createTask(String framework) {
        String taskId = createTask();
        TrainingTask task = tasks.get(taskId);
        if (task != null) {
            task.setFramework(framework);
        }
        return taskId;
    }

    public TrainingTask getTask(String taskId) {
        return tasks.get(taskId);
    }

    public boolean exists(String taskId) {
        return tasks.containsKey(taskId);
    }

    public void removeTask(String taskId) {
        tasks.remove(taskId);
    }

    public Map<String, TrainingTask> getAllTasks() {
        return tasks;
    }

    public int getTaskCount() {
        return tasks.size();
    }

    public int getRunningTaskCount() {
        return (int) tasks.values().stream()
                .filter(TrainingTask::isRunning)
                .count();
    }

    public int getCompletedTaskCount() {
        return (int) tasks.values().stream()
                .filter(TrainingTask::isFinished)
                .count();
    }
}
