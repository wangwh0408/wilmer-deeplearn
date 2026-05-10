package com.example.demo.evaluation.controller;

import com.example.demo.common.Result;
import com.example.demo.evaluation.dto.EvaluationRequest;
import com.example.demo.evaluation.entity.EvaluationLogEntry;
import com.example.demo.evaluation.entity.EvaluationTask;
import com.example.demo.evaluation.service.EvaluationService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Slf4j
@RestController
@RequestMapping("/api/evaluation")
public class EvaluationController {

    @Autowired
    private EvaluationService evaluationService;

    @GetMapping("/info")
    public Result<Map<String, Object>> getApiInfo() {
        Map<String, Object> info = new HashMap<>();
        info.put("name", "FNO2 Model Evaluation API");
        info.put("version", "1.0.0");
        info.put("description", "API for evaluating FNO2 models using ProcessBuilder to call Python evaluation scripts");
        
        Map<String, String> endpoints = new HashMap<>();
        endpoints.put("POST /start", "Start a new evaluation task");
        endpoints.put("GET /status/{taskId}", "Get task status");
        endpoints.put("GET /logs/{taskId}", "Get task logs (supports ?since=N for incremental)");
        endpoints.put("GET /new-logs/{taskId}", "Get new logs since last call (for polling)");
        endpoints.put("POST /cancel/{taskId}", "Cancel a running task");
        endpoints.put("GET /tasks", "Get all tasks");
        endpoints.put("GET /info", "Get this API info");
        info.put("endpoints", endpoints);

        Map<String, String> parameters = new HashMap<>();
        parameters.put("modelPath", "Path to trained model file (e.g., fno2_model.pth)");
        parameters.put("datasetPath", "Root directory of CFD Bench data (e.g., C:/traework/data)");
        parameters.put("samplePercentage", "Percentage of samples to use (0.0 - 100.0, optional)");
        parameters.put("problems", "List of problem types: [cavity, tube, dam, cylinder]");
        parameters.put("categories", "List of categories: [bc, geo, prop]");
        parameters.put("maxCases", "Maximum number of cases per category (optional)");
        parameters.put("normalize", "Whether to normalize data (default: true)");
        parameters.put("device", "Device to use: cpu, cuda (auto-detect by default)");
        parameters.put("outputReportPath", "Path to save evaluation report JSON");
        parameters.put("workingDirectory", "Working directory for the Python process");
        info.put("parameters", parameters);

        Map<String, String> metrics = new HashMap<>();
        metrics.put("MSE", "Mean Squared Error - lower is better");
        metrics.put("RMSE", "Root Mean Squared Error - lower is better");
        metrics.put("MAE", "Mean Absolute Error - lower is better");
        metrics.put("R2", "Coefficient of Determination - 1.0 is perfect");
        metrics.put("MAPE", "Mean Absolute Percentage Error (%) - lower is better");
        info.put("evaluation_metrics", metrics);

        return Result.success(info);
    }

    @PostMapping("/start")
    public Result<Map<String, Object>> startEvaluation(@RequestBody EvaluationRequest request) {
        log.info("[EvaluationController] Starting evaluation task");
        
        String taskId = evaluationService.startEvaluation(request);
        
        Map<String, Object> result = new HashMap<>();
        result.put("taskId", taskId);
        result.put("status", "STARTED");
        result.put("message", "Evaluation task started");
        
        return Result.success(result);
    }

    @GetMapping("/status/{taskId}")
    public Result<Map<String, Object>> getTaskStatus(@PathVariable String taskId) {
        EvaluationTask task = evaluationService.getTaskStatus(taskId);
        
        if (task == null) {
            return Result.error("Task not found: " + taskId);
        }
        
        Map<String, Object> result = new HashMap<>();
        result.put("taskId", task.getTaskId());
        result.put("status", task.getStatus());
        result.put("startTime", task.getStartTime());
        result.put("endTime", task.getEndTime());
        result.put("durationMillis", task.getDurationMillis());
        result.put("exitCode", task.getExitCode());
        result.put("errorMessage", task.getErrorMessage());
        result.put("logCount", task.getLogCount());
        result.put("maxSequence", task.getMaxSequence());
        result.put("isRunning", task.isRunning());
        result.put("isFinished", task.isFinished());
        
        if (task.getMetrics() != null) {
            result.put("metrics", task.getMetrics());
        }
        
        if (task.getModelPath() != null) result.put("modelPath", task.getModelPath());
        if (task.getDatasetPath() != null) result.put("datasetPath", task.getDatasetPath());
        if (task.getSamplePercentage() != null) result.put("samplePercentage", task.getSamplePercentage());
        if (task.getProblems() != null) result.put("problems", task.getProblems());
        if (task.getCategories() != null) result.put("categories", task.getCategories());
        if (task.getMaxCases() != null) result.put("maxCases", task.getMaxCases());
        if (task.getNormalize() != null) result.put("normalize", task.getNormalize());
        if (task.getDevice() != null) result.put("device", task.getDevice());
        
        return Result.success(result);
    }

    @GetMapping("/logs/{taskId}")
    public Result<Map<String, Object>> getLogs(
            @PathVariable String taskId,
            @RequestParam(required = false) Integer since) {
        
        List<EvaluationLogEntry> logs = evaluationService.getLogs(taskId, since);
        
        Map<String, Object> result = new HashMap<>();
        result.put("taskId", taskId);
        result.put("logs", logs);
        result.put("count", logs.size());
        
        EvaluationTask task = evaluationService.getTaskStatus(taskId);
        if (task != null) {
            result.put("taskStatus", task.getStatus());
            result.put("isFinished", task.isFinished());
        }
        
        return Result.success(result);
    }

    @GetMapping("/new-logs/{taskId}")
    public Result<Map<String, Object>> getNewLogs(@PathVariable String taskId) {
        List<EvaluationLogEntry> newLogs = evaluationService.getNewLogs(taskId);
        
        EvaluationTask task = evaluationService.getTaskStatus(taskId);
        
        Map<String, Object> result = new HashMap<>();
        result.put("taskId", taskId);
        result.put("logs", newLogs);
        result.put("count", newLogs.size());
        
        if (task != null) {
            result.put("currentMaxSequence", task.getMaxSequence());
            result.put("taskStatus", task.getStatus());
            result.put("isFinished", task.isFinished());
            if (task.isFinished() && task.getMetrics() != null) {
                result.put("metrics", task.getMetrics());
            }
        }
        
        return Result.success(result);
    }

    @PostMapping("/cancel/{taskId}")
    public Result<Map<String, Object>> cancelEvaluation(@PathVariable String taskId) {
        boolean cancelled = evaluationService.cancelEvaluation(taskId);
        
        if (!cancelled) {
            return Result.error("Failed to cancel task (not running or not found: " + taskId);
        }
        
        Map<String, Object> result = new HashMap<>();
        result.put("taskId", taskId);
        result.put("status", "CANCELLED");
        result.put("message", "Evaluation task cancelled successfully");
        
        return Result.success(result);
    }

    @GetMapping("/tasks")
    public Result<List<EvaluationTask>> getAllTasks() {
        List<EvaluationTask> tasks = evaluationService.getAllTasks();
        return Result.success(tasks);
    }

    @PostMapping("/example")
    public Result<Map<String, Object>> getExampleRequest() {
        Map<String, Object> example = new HashMap<>();
        
        Map<String, Object> minimalRequest = new HashMap<>();
        minimalRequest.put("modelPath", "fno2_model.pth");
        minimalRequest.put("datasetPath", "C:/traework/data");
        example.put("minimalRequest", minimalRequest);
        
        Map<String, Object> fullRequest = new HashMap<>();
        fullRequest.put("modelPath", "fno2_model.pth");
        fullRequest.put("datasetPath", "C:/traework/data");
        fullRequest.put("samplePercentage", 100.0);
        fullRequest.put("problems", java.util.Arrays.asList("cavity", "tube"));
        fullRequest.put("categories", java.util.Arrays.asList("bc", "geo", "prop"));
        fullRequest.put("maxCases", 10);
        fullRequest.put("normalize", true);
        fullRequest.put("device", "cpu");
        fullRequest.put("outputReportPath", "evaluation_report.json");
        fullRequest.put("workingDirectory", "c:/traework/a/wilmer-deeplearn");
        example.put("fullRequest", fullRequest);
        
        Map<String, String> pollingFlow = new HashMap<>();
        pollingFlow.put("1", "POST /api/evaluation/start with your request");
        pollingFlow.put("2", "Get taskId from response");
        pollingFlow.put("3", "Poll GET /api/evaluation/new-logs/{taskId} in a loop");
        pollingFlow.put("4", "Stop when isFinished is true");
        pollingFlow.put("5", "Check metrics are included when task is finished");
        example.put("recommendedPollingFlow", pollingFlow);
        
        return Result.success(example);
    }
}
