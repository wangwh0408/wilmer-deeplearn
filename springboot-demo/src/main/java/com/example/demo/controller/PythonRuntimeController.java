package com.example.demo.controller;

import com.example.demo.common.Result;
import com.example.demo.dto.PythonTrainRequest;
import com.example.demo.entity.PythonTrainResult;
import com.example.demo.service.PythonRuntimeService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@Slf4j
@RestController
@RequestMapping("/api/runtime")
public class PythonRuntimeController {

    @Autowired
    private PythonRuntimeService pythonRuntimeService;

    @GetMapping("/health")
    public Result<Map<String, Object>> health() {
        log.info("[RuntimeController] Health check requested");
        
        Map<String, Object> data = new HashMap<>();
        data.put("status", "ok");
        data.put("service", "PythonRuntimeService");
        data.put("method", "Runtime.getRuntime().exec()");
        data.put("timestamp", System.currentTimeMillis());
        
        return Result.success(data);
    }

    @GetMapping("/info")
    public Result<Map<String, Object>> info() {
        log.info("[RuntimeController] Runtime info requested");
        
        Map<String, Object> info = new HashMap<>();
        info.put("service_name", "PythonRuntimeService");
        info.put("description", "Execute Python scripts using JDK Runtime class");
        info.put("java_version", System.getProperty("java.version"));
        info.put("os_name", System.getProperty("os.name"));
        info.put("user_dir", System.getProperty("user.dir"));
        
        return Result.success(info);
    }

    @PostMapping("/train")
    public Result<PythonTrainResult> train(@RequestBody PythonTrainRequest request) {
        log.info("============================================================");
        log.info("[RuntimeController] Received Runtime train request");
        log.info("============================================================");
        log.info("[RuntimeController] epochs={}, modes={}, width={}", 
                request.getEpochs(), request.getModes(), request.getWidth());
        log.info("[RuntimeController] nTrainSamples={}, resolution={}", 
                request.getNTrainSamples(), request.getResolution());
        log.info("[RuntimeController] quickTrain={}, verbose={}", 
                request.getQuickTrain(), request.getVerbose());

        PythonTrainResult result = pythonRuntimeService.executeWithRuntime(request);

        log.info("============================================================");
        log.info("[RuntimeController] Train request completed");
        log.info("============================================================");
        log.info("[RuntimeController] Success: {}", result.isSuccess());
        log.info("[RuntimeController] Exit Code: {}", result.getExitCode());
        log.info("[RuntimeController] Duration: {} ms", result.getDurationMillis());

        if (result.isSuccess()) {
            return Result.success(200, "Training completed via Runtime", result);
        } else {
            return Result.error(500, "Training failed: " + 
                    (result.getErrorMessage() != null ? result.getErrorMessage() : "Unknown error"));
        }
    }

    @PostMapping("/quick-train/{epochs}")
    public Result<PythonTrainResult> quickTrain(@PathVariable Integer epochs) {
        log.info("============================================================");
        log.info("[RuntimeController] Quick train via Runtime: epochs={}", epochs);
        log.info("============================================================");

        if (epochs == null || epochs <= 0) {
            return Result.error(400, "Invalid epochs value: " + epochs);
        }

        PythonTrainResult result = pythonRuntimeService.quickTrainWithRuntime(epochs);

        log.info("[RuntimeController] Quick train result: success={}, exitCode={}", 
                result.isSuccess(), result.getExitCode());

        if (result.isSuccess()) {
            return Result.success(200, "Quick training completed via Runtime", result);
        } else {
            return Result.error(500, "Quick training failed: " + 
                    (result.getErrorMessage() != null ? result.getErrorMessage() : "Unknown error"));
        }
    }

    @GetMapping("/default-config")
    public Result<PythonTrainRequest> getDefaultConfig() {
        log.info("[RuntimeController] Default config requested");
        
        PythonTrainRequest config = new PythonTrainRequest();
        return Result.success(config);
    }

    @GetMapping("/compare")
    public Result<Map<String, String>> compareMethods() {
        log.info("[RuntimeController] Method comparison requested");
        
        Map<String, String> comparison = new HashMap<>();
        
        comparison.put("ProcessBuilder", 
                "Spring Boot推荐方式，更灵活，支持重定向、环境变量等");
        comparison.put("Runtime.getRuntime().exec()", 
                "JDK传统方式，简单直接，适用于基本场景");
        comparison.put("current_implementation", 
                "Runtime.getRuntime().exec() - 使用独立线程读取输出");
        
        return Result.success(comparison);
    }
}
