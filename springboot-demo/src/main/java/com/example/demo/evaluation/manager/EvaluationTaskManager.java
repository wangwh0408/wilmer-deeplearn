package com.example.demo.evaluation.manager;

import com.example.demo.evaluation.entity.EvaluationTask;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

@Slf4j
@Component
public class EvaluationTaskManager {

    private final Map<String, EvaluationTask> tasks = new ConcurrentHashMap<>();
    private final AtomicLong taskCounter = new AtomicLong(0);

    public String createTask() {
        String taskId = generateTaskId();
        EvaluationTask task = new EvaluationTask(taskId);
        tasks.put(taskId, task);
        log.info("[EvaluationTaskManager] Created evaluation task: {}", taskId);
        return taskId;
    }

    public EvaluationTask getTask(String taskId) {
        return tasks.get(taskId);
    }

    public List<EvaluationTask> getAllTasks() {
        return new ArrayList<>(tasks.values());
    }

    public boolean taskExists(String taskId) {
        return tasks.containsKey(taskId);
    }

    public void removeTask(String taskId) {
        tasks.remove(taskId);
        log.info("[EvaluationTaskManager] Removed evaluation task: {}", taskId);
    }

    public void clearCompletedTasks() {
        List<String> toRemove = new ArrayList<>();
        for (Map.Entry<String, EvaluationTask> entry : tasks.entrySet()) {
            if (entry.getValue().isFinished()) {
                toRemove.add(entry.getKey());
            }
        }
        for (String taskId : toRemove) {
            removeTask(taskId);
        }
    }

    private String generateTaskId() {
        long counter = taskCounter.incrementAndGet();
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMddHHmmss"));
        return "EVAL_" + timestamp + "_" + String.format("%06d", counter);
    }
}
