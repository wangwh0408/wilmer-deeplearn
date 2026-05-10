package com.example.demo.controller;

import com.example.demo.entity.ModelServerEntity;
import com.example.demo.service.ModelServerService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

@Slf4j
@RestController
@RequestMapping("/api/model-server")
public class ModelServerController {

    @Autowired
    private ModelServerService modelServerService;

    @PostMapping("/start")
    public ResponseEntity<Map<String, Object>> startServer(
            @RequestBody(required = false) Map<String, Object> request) {

        Map<String, Object> response = new HashMap<>();

        try {
            String modelPath = request != null ?
                    (String) request.getOrDefault("model_path", "fno2_full_model.pth") :
                    "fno2_full_model.pth";

            String framework = request != null ?
                    (String) request.getOrDefault("framework", "auto") :
                    "auto";

            String host = request != null ?
                    (String) request.getOrDefault("host", "0.0.0.0") :
                    "0.0.0.0";

            Integer port = request != null ?
                    (Integer) request.get("port") :
                    null;

            String workingDir = request != null ?
                    (String) request.get("working_dir") :
                    null;

            ModelServerEntity.ServerType serverType = null;
            if (request != null && request.get("server_type") != null) {
                String serverTypeStr = ((String) request.get("server_type")).toUpperCase();
                try {
                    serverType = ModelServerEntity.ServerType.valueOf(serverTypeStr);
                } catch (IllegalArgumentException e) {
                    throw new IllegalArgumentException("Invalid server_type: " + serverTypeStr + 
                            ". Valid values: FLASK, FASTAPI");
                }
            }

            log.info("[ModelServerController] Starting server - model: {}, framework: {}, server_type: {}, port: {}",
                    modelPath, framework, serverType, port);

            String serverId;
            if (serverType != null) {
                serverId = modelServerService.startServer(
                        modelPath, framework, host, port, workingDir, serverType);
            } else {
                serverId = modelServerService.startServer(
                        modelPath, framework, host, port, workingDir);
            }

            Optional<ModelServerEntity> server = modelServerService.getServer(serverId);

            response.put("success", true);
            response.put("server_id", serverId);
            response.put("message", "Server started successfully");

            if (server.isPresent()) {
                ModelServerEntity entity = server.get();
                response.put("server", toMap(entity));
            }

            return ResponseEntity.ok(response);

        } catch (Exception e) {
            log.error("[ModelServerController] Failed to start server: {}", e.getMessage(), e);
            response.put("success", false);
            response.put("error", e.getMessage());
            return ResponseEntity.status(500).body(response);
        }
    }

    @PostMapping("/{serverId}/stop")
    public ResponseEntity<Map<String, Object>> stopServer(@PathVariable String serverId) {
        Map<String, Object> response = new HashMap<>();

        try {
            log.info("[ModelServerController] Stopping server: {}", serverId);

            boolean stopped = modelServerService.stopServer(serverId);

            Optional<ModelServerEntity> server = modelServerService.getServer(serverId);

            response.put("success", stopped);
            response.put("server_id", serverId);
            response.put("message", stopped ? "Server stopped successfully" : "Failed to stop server");

            if (server.isPresent()) {
                response.put("server", toMap(server.get()));
            }

            return ResponseEntity.ok(response);

        } catch (Exception e) {
            log.error("[ModelServerController] Failed to stop server {}: {}", serverId, e.getMessage(), e);
            response.put("success", false);
            response.put("error", e.getMessage());
            return ResponseEntity.status(500).body(response);
        }
    }

    @PostMapping("/stop-all")
    public ResponseEntity<Map<String, Object>> stopAllServers() {
        Map<String, Object> response = new HashMap<>();

        try {
            log.info("[ModelServerController] Stopping all servers");

            boolean allStopped = modelServerService.stopAllServers();

            response.put("success", allStopped);
            response.put("message", allStopped ? "All servers stopped" : "Some servers failed to stop");

            List<ModelServerEntity> servers = modelServerService.getAllServers();
            response.put("servers", servers.stream().map(this::toMap).toArray());

            return ResponseEntity.ok(response);

        } catch (Exception e) {
            log.error("[ModelServerController] Failed to stop all servers: {}", e.getMessage(), e);
            response.put("success", false);
            response.put("error", e.getMessage());
            return ResponseEntity.status(500).body(response);
        }
    }

    @GetMapping("/{serverId}")
    public ResponseEntity<Map<String, Object>> getServer(@PathVariable String serverId) {
        Map<String, Object> response = new HashMap<>();

        try {
            Optional<ModelServerEntity> server = modelServerService.getServer(serverId);

            if (server.isPresent()) {
                response.put("success", true);
                response.put("server", toMap(server.get()));
                return ResponseEntity.ok(response);
            } else {
                response.put("success", false);
                response.put("error", "Server not found: " + serverId);
                return ResponseEntity.status(404).body(response);
            }

        } catch (Exception e) {
            log.error("[ModelServerController] Failed to get server {}: {}", serverId, e.getMessage(), e);
            response.put("success", false);
            response.put("error", e.getMessage());
            return ResponseEntity.status(500).body(response);
        }
    }

    @GetMapping("/list")
    public ResponseEntity<Map<String, Object>> listServers() {
        Map<String, Object> response = new HashMap<>();

        try {
            List<ModelServerEntity> servers = modelServerService.getAllServers();

            response.put("success", true);
            response.put("count", servers.size());
            response.put("servers", servers.stream().map(this::toMap).toArray());

            return ResponseEntity.ok(response);

        } catch (Exception e) {
            log.error("[ModelServerController] Failed to list servers: {}", e.getMessage(), e);
            response.put("success", false);
            response.put("error", e.getMessage());
            return ResponseEntity.status(500).body(response);
        }
    }

    @GetMapping("/running")
    public ResponseEntity<Map<String, Object>> listRunningServers() {
        Map<String, Object> response = new HashMap<>();

        try {
            List<ModelServerEntity> servers = modelServerService.getRunningServers();

            response.put("success", true);
            response.put("count", servers.size());
            response.put("servers", servers.stream().map(this::toMap).toArray());

            return ResponseEntity.ok(response);

        } catch (Exception e) {
            log.error("[ModelServerController] Failed to list running servers: {}", e.getMessage(), e);
            response.put("success", false);
            response.put("error", e.getMessage());
            return ResponseEntity.status(500).body(response);
        }
    }

    @GetMapping("/{serverId}/logs")
    public ResponseEntity<Map<String, Object>> getServerLogs(@PathVariable String serverId) {
        Map<String, Object> response = new HashMap<>();

        try {
            List<String> logs = modelServerService.getServerLogs(serverId);

            response.put("success", true);
            response.put("server_id", serverId);
            response.put("log_count", logs.size());
            response.put("logs", logs);

            return ResponseEntity.ok(response);

        } catch (Exception e) {
            log.error("[ModelServerController] Failed to get logs for server {}: {}", serverId, e.getMessage(), e);
            response.put("success", false);
            response.put("error", e.getMessage());
            return ResponseEntity.status(500).body(response);
        }
    }

    @GetMapping("/{serverId}/health")
    public ResponseEntity<Map<String, Object>> healthCheck(@PathVariable String serverId) {
        Map<String, Object> response = new HashMap<>();

        try {
            Map<String, Object> result = modelServerService.callHealthCheck(serverId);

            response.put("success", true);
            response.put("server_id", serverId);
            response.put("health", result);

            return ResponseEntity.ok(response);

        } catch (Exception e) {
            log.error("[ModelServerController] Health check failed for server {}: {}", serverId, e.getMessage(), e);
            response.put("success", false);
            response.put("error", e.getMessage());
            return ResponseEntity.status(500).body(response);
        }
    }

    @GetMapping("/{serverId}/info")
    public ResponseEntity<Map<String, Object>> getServerInfo(@PathVariable String serverId) {
        Map<String, Object> response = new HashMap<>();

        try {
            Map<String, Object> result = modelServerService.getServerInfo(serverId);

            response.put("success", true);
            response.put("server_id", serverId);
            response.put("info", result);

            return ResponseEntity.ok(response);

        } catch (Exception e) {
            log.error("[ModelServerController] Failed to get info for server {}: {}", serverId, e.getMessage(), e);
            response.put("success", false);
            response.put("error", e.getMessage());
            return ResponseEntity.status(500).body(response);
        }
    }

    @PostMapping("/{serverId}/predict")
    public ResponseEntity<Map<String, Object>> predict(
            @PathVariable String serverId,
            @RequestBody Map<String, Object> request) {

        Map<String, Object> response = new HashMap<>();

        try {
            log.info("[ModelServerController] Predict request for server: {}", serverId);

            Map<String, Object> result = modelServerService.predict(serverId, request);

            response.put("success", true);
            response.put("server_id", serverId);
            response.put("result", result);

            return ResponseEntity.ok(response);

        } catch (Exception e) {
            log.error("[ModelServerController] Prediction failed for server {}: {}", serverId, e.getMessage(), e);
            response.put("success", false);
            response.put("error", e.getMessage());
            return ResponseEntity.status(500).body(response);
        }
    }

    @PostMapping("/{serverId}/reload")
    public ResponseEntity<Map<String, Object>> reloadModel(
            @PathVariable String serverId,
            @RequestBody(required = false) Map<String, Object> request) {

        Map<String, Object> response = new HashMap<>();

        try {
            String modelPath = request != null ? (String) request.get("model_path") : null;
            String framework = request != null ? (String) request.get("framework") : null;

            log.info("[ModelServerController] Reload model for server: {}, model: {}", serverId, modelPath);

            Map<String, Object> result = modelServerService.reloadModel(serverId, modelPath, framework);

            response.put("success", true);
            response.put("server_id", serverId);
            response.put("result", result);

            return ResponseEntity.ok(response);

        } catch (Exception e) {
            log.error("[ModelServerController] Failed to reload model for server {}: {}", serverId, e.getMessage(), e);
            response.put("success", false);
            response.put("error", e.getMessage());
            return ResponseEntity.status(500).body(response);
        }
    }

    private Map<String, Object> toMap(ModelServerEntity entity) {
        Map<String, Object> map = new HashMap<>();
        map.put("id", entity.getId());
        map.put("server_id", entity.getServerId());
        map.put("status", entity.getStatus() != null ? entity.getStatus().name() : null);
        map.put("model_path", entity.getModelPath());
        map.put("framework", entity.getFramework());
        map.put("server_type", entity.getServerType() != null ? entity.getServerType().name() : null);
        map.put("port", entity.getPort());
        map.put("host", entity.getHost());
        map.put("pid", entity.getPid());
        map.put("command", entity.getCommand());
        map.put("device", entity.getDevice());
        map.put("start_time", entity.getStartTime() != null ? entity.getStartTime().toString() : null);
        map.put("stop_time", entity.getStopTime() != null ? entity.getStopTime().toString() : null);
        map.put("error_message", entity.getErrorMessage());
        map.put("create_time", entity.getCreateTime() != null ? entity.getCreateTime().toString() : null);
        map.put("update_time", entity.getUpdateTime() != null ? entity.getUpdateTime().toString() : null);
        return map;
    }
}
