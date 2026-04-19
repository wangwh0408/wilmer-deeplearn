package com.example.demo.controller;

import com.example.demo.common.Result;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api")
public class HelloController {

    @GetMapping("/hello")
    public Result<Map<String, Object>> hello() {
        Map<String, Object> data = new HashMap<>();
        data.put("message", "Hello, Spring Boot!");
        data.put("timestamp", System.currentTimeMillis());
        data.put("status", "ok");
        return Result.success(data);
    }

    @GetMapping("/health")
    public Result<String> health() {
        return Result.success("Application is running!");
    }

    @GetMapping("/info")
    public Result<Map<String, Object>> info() {
        Map<String, Object> info = new HashMap<>();
        info.put("app", "Spring Boot Demo");
        info.put("version", "1.0.0");
        info.put("framework", "Spring Boot 2.7.18");
        info.put("java", System.getProperty("java.version"));
        return Result.success(info);
    }
}
