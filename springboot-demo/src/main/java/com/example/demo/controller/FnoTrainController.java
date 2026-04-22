package com.example.demo.controller;

import com.example.demo.common.Result;
import com.example.demo.dto.FnoTrainRequest;
import com.example.demo.entity.TrainingLogEntry;
import com.example.demo.entity.TrainingTask;
import com.example.demo.service.FnoTrainService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

@Slf4j
@RestController
@RequestMapping("/api/fno/train")
public class FnoTrainController {

    @Autowired
    private FnoTrainService fnoTrainService;

    @PostMapping("/start")
    public Result<Map<String, Object>> startTraining(@RequestBody FnoTrainRequest request) {
        log.info("[FnoTrainController] Starting FNO training request");
        log.info("[FnoTrainController] Framework: {}", request.getFramework());
        log.info("[FnoTrainController] Epochs: {}", request.getEpochs());
        log.info("[FnoTrainController] Batch size: {}", request.getBatchSize());

        String taskId = fnoTrainService.startTraining(request);

        Map<String, Object> response = new HashMap<>();
        response.put("taskId", taskId);
        response.put("message", "Training started successfully");
        response.put("status", "STARTED");

        log.info("[FnoTrainController] Training task created: {}", taskId);

        return Result.success(response);
    }

    @PostMapping("/quick-test")
    public Result<Map<String, Object>> quickTestTraining(
            @RequestParam(required = false, defaultValue = "pytorch") String framework) {
        log.info("[FnoTrainController] Quick test training with framework: {}", framework);

        String taskId;
        if ("paddle".equalsIgnoreCase(framework)) {
            taskId = fnoTrainService.quickTestTrainingPaddle();
        } else {
            taskId = fnoTrainService.quickTestTraining();
        }

        Map<String, Object> response = new HashMap<>();
        response.put("taskId", taskId);
        response.put("message", "Quick test training started");
        response.put("framework", framework);

        return Result.success(response);
    }

    @GetMapping("/status/{taskId}")
    public Result<TrainingTask> getTaskStatus(@PathVariable String taskId) {
        log.info("[FnoTrainController] Getting status for task: {}", taskId);

        TrainingTask task = fnoTrainService.getTaskStatus(taskId);

        if (task == null) {
            return Result.error("Task not found: " + taskId);
        }

        return Result.success(task);
    }

    @GetMapping("/logs/{taskId}")
    public Result<List<TrainingLogEntry>> getLogs(
            @PathVariable String taskId,
            @RequestParam(required = false) Integer since) {
        log.info("[FnoTrainController] Getting logs for task: {} (since: {})", taskId, since);

        List<TrainingLogEntry> logs = fnoTrainService.getLogs(taskId, since);

        return Result.success(logs);
    }

    @PostMapping("/cancel/{taskId}")
    public Result<Map<String, Object>> cancelTraining(@PathVariable String taskId) {
        log.info("[FnoTrainController] Cancelling training task: {}", taskId);

        boolean cancelled = fnoTrainService.cancelTraining(taskId);

        Map<String, Object> response = new HashMap<>();
        response.put("taskId", taskId);

        if (cancelled) {
            response.put("message", "Training cancelled successfully");
            response.put("cancelled", true);
            return Result.success(response);
        } else {
            response.put("message", "Task not running or not found");
            response.put("cancelled", false);
            return Result.error("Task not running or not found: " + taskId);
        }
    }

    @GetMapping("/tasks")
    public Result<List<TrainingTask>> getAllTasks() {
        log.info("[FnoTrainController] Getting all tasks");

        List<TrainingTask> tasks = fnoTrainService.getAllTasks();

        return Result.success(tasks);
    }

    @GetMapping("/info")
    public Result<Map<String, Object>> getInfo() {
        log.info("[FnoTrainController] Getting API info");

        Map<String, Object> info = new HashMap<>();

        info.put("supportedFrameworks", Arrays.asList("pytorch", "torch", "paddle", "paddlepaddle"));

        Map<String, Object> defaultConfig = new HashMap<>();
        defaultConfig.put("framework", "pytorch");
        defaultConfig.put("epochs", 100);
        defaultConfig.put("batchSize", 8);
        defaultConfig.put("learningRate", 0.001);
        defaultConfig.put("modes1", 12);
        defaultConfig.put("modes2", 12);
        defaultConfig.put("width", 32);
        defaultConfig.put("nLayers", 4);
        info.put("defaultConfig", defaultConfig);

        Map<String, Object> endpoints = new HashMap<>();
        endpoints.put("POST /api/fno/train/start", "Start a new training task");
        endpoints.put("POST /api/fno/train/quick-test", "Quick test training (3 epochs, 5 cases)");
        endpoints.put("GET /api/fno/train/status/{taskId}", "Get task status");
        endpoints.put("GET /api/fno/train/logs/{taskId}", "Get task logs (add ?since=N to get logs after index N)");
        endpoints.put("POST /api/fno/train/cancel/{taskId}", "Cancel a running task");
        endpoints.put("GET /api/fno/train/tasks", "Get all tasks");
        endpoints.put("GET /api/fno/train/info", "Get API information");
        info.put("endpoints", endpoints);

        return Result.success(info);
    }

    @GetMapping("/example")
    public Result<Map<String, Object>> getExampleRequest() {
        Map<String, Object> example = new HashMap<>();

        Map<String, Object> pytorchExample = new HashMap<>();
        pytorchExample.put("framework", "pytorch");
        pytorchExample.put("dataRoot", "/path/to/cfdbench/data");
        pytorchExample.put("problems", Arrays.asList("cavity"));
        pytorchExample.put("categories", Arrays.asList("bc", "prop"));
        pytorchExample.put("maxCases", 10);
        pytorchExample.put("epochs", 100);
        pytorchExample.put("batchSize", 8);
        pytorchExample.put("learningRate", 0.001);
        pytorchExample.put("modes1", 12);
        pytorchExample.put("modes2", 12);
        pytorchExample.put("width", 32);
        pytorchExample.put("nLayers", 4);
        pytorchExample.put("seed", 42);
        pytorchExample.put("verbose", true);

        Map<String, Object> paddleExample = new HashMap<>();
        paddleExample.put("framework", "paddle");
        paddleExample.put("dataRoot", "/path/to/cfdbench/data");
        paddleExample.put("problems", Arrays.asList("cavity", "cylinder"));
        paddleExample.put("categories", Arrays.asList("prop"));
        paddleExample.put("maxCases", 20);
        paddleExample.put("epochs", 50);
        paddleExample.put("batchSize", 4);
        paddleExample.put("learningRate", 0.001);
        paddleExample.put("modes1", 8);
        paddleExample.put("modes2", 8);
        paddleExample.put("width", 16);
        paddleExample.put("nLayers", 2);
        paddleExample.put("seed", 123);
        paddleExample.put("verbose", true);

        example.put("pytorch", pytorchExample);
        example.put("paddle", paddleExample);

        Map<String, Object> curlExample = new HashMap<>();
        curlExample.put("startTraining", "curl -X POST http://localhost:8080/api/fno/train/start \\\n" +
                "  -H \"Content-Type: application/json\" \\\n" +
                "  -d '{\"framework\": \"pytorch\", \"epochs\": 100, \"maxCases\": 10}'");

        curlExample.put("getStatus", "curl http://localhost:8080/api/fno/train/status/<taskId>");
        curlExample.put("getLogs", "curl http://localhost:8080/api/fno/train/logs/<taskId>");
        curlExample.put("getLogsSince", "curl http://localhost:8080/api/fno/train/logs/<taskId>?since=100");
        curlExample.put("cancelTask", "curl -X POST http://localhost:8080/api/fno/train/cancel/<taskId>");

        example.put("curlExamples", curlExample);

        return Result.success(example);
    }
}
