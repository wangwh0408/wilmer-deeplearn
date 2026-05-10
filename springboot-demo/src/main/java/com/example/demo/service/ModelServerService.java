package com.example.demo.service;

import com.example.demo.config.PythonTrainConfig;
import com.example.demo.entity.ModelServerEntity;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import javax.annotation.PostConstruct;
import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

@Slf4j
@Service
public class ModelServerService {

    @Autowired
    private PythonTrainConfig config;

    private String defaultWorkingDir;
    private String pythonExecutable;

    private final ExecutorService executorService = Executors.newCachedThreadPool();
    private final Map<String, Process> runningServers = new ConcurrentHashMap<>();
    private final Map<String, Future<?>> serverFutures = new ConcurrentHashMap<>();
    private final Map<String, List<String>> serverLogs = new ConcurrentHashMap<>();
    private final Map<Integer, String> portToServerId = new ConcurrentHashMap<>();

    private final Map<String, ModelServerEntity> serverStore = new ConcurrentHashMap<>();
    private final AtomicLong idCounter = new AtomicLong(1);

    private final RestTemplate restTemplate = new RestTemplate();

    private static final String DEFAULT_HOST = "0.0.0.0";
    private static final int DEFAULT_PORT_FLASK = 5000;
    private static final int DEFAULT_PORT_FASTAPI = 8000;
    private static final int HEALTH_CHECK_TIMEOUT_MS = 30000;
    private static final int HEALTH_CHECK_INTERVAL_MS = 1000;

    @PostConstruct
    public void init() {
        if (config.getWorkingDirectory() == null || config.getWorkingDirectory().isEmpty()) {
            String userDir = System.getProperty("user.dir");
            File parentDir = new File(userDir).getParentFile();
            defaultWorkingDir = parentDir != null ? parentDir.getAbsolutePath() : userDir;
        } else {
            defaultWorkingDir = config.getWorkingDirectory();
        }

        pythonExecutable = config.getPythonExecutable() != null ?
                config.getPythonExecutable() : "python";

        log.info("[ModelServerService] Default working directory: {}", defaultWorkingDir);
        log.info("[ModelServerService] Python executable: {}", pythonExecutable);
    }

    public String generateServerId() {
        return "ms-" + UUID.randomUUID().toString().replace("-", "").substring(0, 12);
    }

    public int getAvailablePort(int startPort) {
        int port = startPort;
        while (isPortInUse(port)) {
            port++;
        }
        return port;
    }

    private boolean isPortInUse(int port) {
        if (portToServerId.containsKey(port)) {
            ModelServerEntity server = serverStore.get(portToServerId.get(port));
            return server != null &&
                    (server.getStatus() == ModelServerEntity.Status.RUNNING ||
                     server.getStatus() == ModelServerEntity.Status.STARTING);
        }
        return false;
    }

    public String startServer(String modelPath, String framework, String host, Integer port, String workingDir) {
        return startServer(modelPath, framework, host, port, workingDir, null);
    }

    public String startServer(String modelPath, String framework, String host, Integer port, String workingDir, 
            ModelServerEntity.ServerType serverType) {
        
        ModelServerEntity.ServerType actualServerType = serverType != null ? 
                serverType : ModelServerEntity.ServerType.FLASK;
        
        int defaultPort = actualServerType == ModelServerEntity.ServerType.FASTAPI ? 
                DEFAULT_PORT_FASTAPI : DEFAULT_PORT_FLASK;
        
        String actualWorkingDir = (workingDir != null && !workingDir.isEmpty()) ? workingDir : defaultWorkingDir;
        String actualHost = (host != null && !host.isEmpty()) ? host : DEFAULT_HOST;
        int actualPort = (port != null) ? getAvailablePort(port) : getAvailablePort(defaultPort);

        String serverId = generateServerId();

        File modelFile = new File(actualWorkingDir, modelPath);
        if (!modelFile.exists()) {
            throw new IllegalArgumentException("Model file not found: " + modelFile.getAbsolutePath());
        }

        String actualFramework = (framework != null && !framework.isEmpty()) ?
                framework : detectFramework(modelPath);

        List<String> command = buildServerCommand(
                modelFile.getAbsolutePath(),
                actualFramework,
                actualHost,
                actualPort,
                actualServerType
        );

        String commandStr = String.join(" ", command);

        ModelServerEntity entity = ModelServerEntity.builder()
                .id(idCounter.getAndIncrement())
                .serverId(serverId)
                .status(ModelServerEntity.Status.STARTING)
                .modelPath(modelPath)
                .framework(actualFramework)
                .port(actualPort)
                .host(actualHost)
                .serverType(actualServerType)
                .pid(0)
                .command(commandStr)
                .startTime(LocalDateTime.now())
                .build();
        entity.onCreate();
        serverStore.put(serverId, entity);

        serverLogs.put(serverId, new ArrayList<>());
        addServerLog(serverId, "Starting model server...");
        addServerLog(serverId, "Server ID: " + serverId);
        addServerLog(serverId, "Model path: " + modelPath);
        addServerLog(serverId, "Framework: " + actualFramework);
        addServerLog(serverId, "Server type: " + actualServerType);
        addServerLog(serverId, "Host: " + actualHost);
        addServerLog(serverId, "Port: " + actualPort);
        addServerLog(serverId, "Working directory: " + actualWorkingDir);
        addServerLog(serverId, "Command: " + commandStr);

        try {
            ProcessBuilder processBuilder = new ProcessBuilder(command);
            processBuilder.directory(new File(actualWorkingDir));
            processBuilder.redirectErrorStream(true);

            Map<String, String> env = processBuilder.environment();
            env.put("PYTHONUNBUFFERED", "1");
            env.put("PYTHONDONTWRITEBYTECODE", "1");

            addServerLog(serverId, "Starting process...");
            Process process = processBuilder.start();

            int pid = getProcessId(process);
            entity.setPid(pid);
            entity.setStatus(ModelServerEntity.Status.STARTING);
            entity.onUpdate();

            runningServers.put(serverId, process);
            portToServerId.put(actualPort, serverId);

            Future<?> future = executorService.submit(() -> {
                try {
                    monitorServerProcess(serverId, process);
                } catch (Exception e) {
                    log.error("[ModelServerService] Error monitoring server {}: {}", serverId, e.getMessage());
                    addServerLog(serverId, "Error: " + e.getMessage());
                    updateServerStatus(serverId, ModelServerEntity.Status.FAILED, e.getMessage());
                } finally {
                    runningServers.remove(serverId);
                    serverFutures.remove(serverId);
                    portToServerId.remove(actualPort);
                }
            });

            serverFutures.put(serverId, future);

            addServerLog(serverId, "Process started, PID: " + pid);
            addServerLog(serverId, "Waiting for server to be ready...");

            boolean healthCheckPassed = waitForHealthCheck(actualHost, actualPort, serverId);

            if (healthCheckPassed) {
                addServerLog(serverId, "Server is ready!");
                addServerLog(serverId, "Health check passed");
                updateServerStatus(serverId, ModelServerEntity.Status.RUNNING, null);

                try {
                    Map<String, Object> info = getServerInfo(serverId);
                    if (info != null && info.get("device") != null) {
                        entity.setDevice(info.get("device").toString());
                        entity.onUpdate();
                    }
                } catch (Exception e) {
                    log.warn("Failed to get server info: {}", e.getMessage());
                }
            } else {
                addServerLog(serverId, "WARNING: Health check timed out");
                addServerLog(serverId, "Server may still be starting or failed to start");
                updateServerStatus(serverId, ModelServerEntity.Status.FAILED, "Health check timed out");
            }

            return serverId;

        } catch (Exception e) {
            log.error("[ModelServerService] Failed to start server {}: {}", serverId, e.getMessage());
            addServerLog(serverId, "Failed to start: " + e.getMessage());
            updateServerStatus(serverId, ModelServerEntity.Status.FAILED, e.getMessage());
            throw new RuntimeException("Failed to start model server: " + e.getMessage(), e);
        }
    }

    private List<String> buildServerCommand(String modelPath, String framework, String host, int port) {
        return buildServerCommand(modelPath, framework, host, port, ModelServerEntity.ServerType.FLASK);
    }

    private List<String> buildServerCommand(String modelPath, String framework, String host, int port,
            ModelServerEntity.ServerType serverType) {
        List<String> command = new ArrayList<>();

        command.add(pythonExecutable);
        command.add("-u");

        if (serverType == ModelServerEntity.ServerType.FASTAPI) {
            command.add("model_server_fastapi.py");
        } else {
            command.add("model_server.py");
        }

        command.add("--model_path");
        command.add(modelPath);

        if (framework != null && !framework.isEmpty() && !"auto".equals(framework)) {
            command.add("--framework");
            command.add(framework);
        }

        command.add("--host");
        command.add(host);

        command.add("--port");
        command.add(String.valueOf(port));

        return command;
    }

    private String detectFramework(String modelPath) {
        String lowerPath = modelPath.toLowerCase();
        if (lowerPath.contains("paddle")) {
            return "paddle";
        }
        return "pytorch";
    }

    private int getProcessId(Process process) {
        try {
            java.lang.reflect.Field pidField = process.getClass().getDeclaredField("pid");
            pidField.setAccessible(true);
            return pidField.getInt(process);
        } catch (NoSuchFieldException e) {
            try {
                java.lang.reflect.Field handleField = process.getClass().getDeclaredField("handle");
                handleField.setAccessible(true);
                long handle = handleField.getLong(process);
                return (int) handle;
            } catch (Exception e2) {
                log.debug("Failed to get process ID from fields: {}", e2.getMessage());
            }
        } catch (Exception e) {
            log.debug("Failed to get process ID from pid field: {}", e.getMessage());
        }
        
        try {
            String processStr = process.toString();
            java.util.regex.Pattern pattern = java.util.regex.Pattern.compile("pid=(\\d+)");
            java.util.regex.Matcher matcher = pattern.matcher(processStr);
            if (matcher.find()) {
                return Integer.parseInt(matcher.group(1));
            }
        } catch (Exception e) {
            log.debug("Failed to parse process ID from toString(): {}", e.getMessage());
        }
        
        return -1;
    }

    private void monitorServerProcess(String serverId, Process process) {
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {

            String line;
            while ((line = reader.readLine()) != null) {
                addServerLog(serverId, line);
                log.debug("[ModelServer-{}] {}", serverId, line);
            }

            int exitCode = process.waitFor();
            addServerLog(serverId, "Process exited with code: " + exitCode);

            ModelServerEntity entity = serverStore.get(serverId);
            if (entity != null) {
                if (entity.getStatus() == ModelServerEntity.Status.RUNNING ||
                    entity.getStatus() == ModelServerEntity.Status.STARTING) {
                    if (exitCode == 0) {
                        addServerLog(serverId, "Server stopped normally");
                        updateServerStatus(serverId, ModelServerEntity.Status.STOPPED, null);
                    } else {
                        addServerLog(serverId, "Server stopped with error code: " + exitCode);
                        updateServerStatus(serverId, ModelServerEntity.Status.FAILED,
                                "Process exited with code: " + exitCode);
                    }
                }
            }

        } catch (Exception e) {
            log.error("[ModelServerService] Error reading output for server {}: {}", serverId, e.getMessage());
            addServerLog(serverId, "Error reading output: " + e.getMessage());
        }
    }

    private boolean waitForHealthCheck(String host, int port, String serverId) {
        String checkHost = "0.0.0.0".equals(host) ? "127.0.0.1" : host;
        String healthUrl = "http://" + checkHost + ":" + port + "/health";

        long startTime = System.currentTimeMillis();

        while (System.currentTimeMillis() - startTime < HEALTH_CHECK_TIMEOUT_MS) {
            try {
                Thread.sleep(HEALTH_CHECK_INTERVAL_MS);

                ResponseEntity<Map> response = restTemplate.getForEntity(healthUrl, Map.class);

                if (response.getStatusCode() == HttpStatus.OK) {
                    Map<String, Object> body = response.getBody();
                    if (body != null && Boolean.TRUE.equals(body.get("model_loaded"))) {
                        addServerLog(serverId, "Health check passed: model loaded");
                        return true;
                    } else if (body != null && "healthy".equals(body.get("status"))) {
                        addServerLog(serverId, "Health check passed: server running");
                        return true;
                    }
                }

            } catch (Exception e) {
                log.debug("Health check failed ({}ms elapsed): {}",
                        System.currentTimeMillis() - startTime, e.getMessage());
            }
        }

        return false;
    }

    private void addServerLog(String serverId, String message) {
        List<String> logs = serverLogs.computeIfAbsent(serverId, k -> new ArrayList<>());
        String timestamp = java.time.LocalTime.now().toString();
        logs.add("[" + timestamp + "] " + message);

        if (logs.size() > 10000) {
            logs = logs.subList(logs.size() - 5000, logs.size());
            serverLogs.put(serverId, logs);
        }
    }

    public void updateServerStatus(String serverId, ModelServerEntity.Status status, String errorMessage) {
        ModelServerEntity entity = serverStore.get(serverId);
        if (entity != null) {
            entity.setStatus(status);

            if (errorMessage != null) {
                entity.setErrorMessage(errorMessage);
            }

            if (status == ModelServerEntity.Status.STOPPED ||
                status == ModelServerEntity.Status.FAILED) {
                entity.setStopTime(LocalDateTime.now());
            }

            entity.onUpdate();
        }
    }

    public boolean stopServer(String serverId) {
        ModelServerEntity entity = serverStore.get(serverId);
        if (entity == null) {
            log.warn("[ModelServerService] Server not found: {}", serverId);
            return false;
        }

        if (entity.getStatus() == ModelServerEntity.Status.STOPPED ||
            entity.getStatus() == ModelServerEntity.Status.FAILED) {
            log.info("[ModelServerService] Server {} is already stopped", serverId);
            return true;
        }

        addServerLog(serverId, "Stopping server...");
        entity.setStatus(ModelServerEntity.Status.STOPPING);
        entity.onUpdate();

        try {
            String host = "0.0.0.0".equals(entity.getHost()) ? "127.0.0.1" : entity.getHost();
            String shutdownUrl = "http://" + host + ":" + entity.getPort() + "/shutdown";

            addServerLog(serverId, "Sending shutdown request to: " + shutdownUrl);

            try {
                restTemplate.postForEntity(shutdownUrl, null, Map.class);
                addServerLog(serverId, "Shutdown request sent");
            } catch (Exception e) {
                addServerLog(serverId, "Shutdown request failed (expected during shutdown): " + e.getMessage());
            }

            Process process = runningServers.get(serverId);
            if (process != null) {
                addServerLog(serverId, "Waiting for process to exit...");

                boolean exited = false;
                for (int i = 0; i < 30; i++) {
                    try {
                        int exitCode = process.exitValue();
                        exited = true;
                        addServerLog(serverId, "Process exited with code: " + exitCode);
                        break;
                    } catch (IllegalThreadStateException e) {
                        Thread.sleep(200);
                    }
                }

                if (!exited) {
                    addServerLog(serverId, "Process did not exit gracefully, forcing termination...");
                    process.destroyForcibly();
                    addServerLog(serverId, "Process forcibly terminated");
                }
            }

            updateServerStatus(serverId, ModelServerEntity.Status.STOPPED, null);
            addServerLog(serverId, "Server stopped successfully");

            return true;

        } catch (Exception e) {
            log.error("[ModelServerService] Error stopping server {}: {}", serverId, e.getMessage());
            addServerLog(serverId, "Error stopping server: " + e.getMessage());

            Process process = runningServers.get(serverId);
            if (process != null) {
                process.destroyForcibly();
            }

            updateServerStatus(serverId, ModelServerEntity.Status.FAILED,
                    "Error during shutdown: " + e.getMessage());

            return false;
        }
    }

    public boolean stopAllServers() {
        List<ModelServerEntity> runningServersList = serverStore.values().stream()
                .filter(s -> s.getStatus() == ModelServerEntity.Status.RUNNING)
                .collect(Collectors.toList());

        boolean allStopped = true;
        for (ModelServerEntity server : runningServersList) {
            boolean stopped = stopServer(server.getServerId());
            if (!stopped) {
                allStopped = false;
            }
        }

        return allStopped;
    }

    public Optional<ModelServerEntity> getServer(String serverId) {
        return Optional.ofNullable(serverStore.get(serverId));
    }

    public Optional<ModelServerEntity> getServerByPort(int port) {
        String serverId = portToServerId.get(port);
        if (serverId != null) {
            return Optional.ofNullable(serverStore.get(serverId));
        }
        return Optional.empty();
    }

    public List<ModelServerEntity> getAllServers() {
        return serverStore.values().stream()
                .sorted(Comparator.comparing(ModelServerEntity::getCreateTime).reversed())
                .collect(Collectors.toList());
    }

    public List<ModelServerEntity> getRunningServers() {
        return serverStore.values().stream()
                .filter(s -> s.getStatus() == ModelServerEntity.Status.RUNNING)
                .sorted(Comparator.comparing(ModelServerEntity::getCreateTime).reversed())
                .collect(Collectors.toList());
    }

    public List<String> getServerLogs(String serverId) {
        return serverLogs.getOrDefault(serverId, new ArrayList<>());
    }

    public Map<String, Object> getServerInfo(String serverId) {
        ModelServerEntity entity = serverStore.get(serverId);
        if (entity == null) {
            throw new IllegalArgumentException("Server not found: " + serverId);
        }

        if (entity.getStatus() != ModelServerEntity.Status.RUNNING) {
            throw new IllegalStateException("Server is not running: " + entity.getStatus());
        }

        String host = "0.0.0.0".equals(entity.getHost()) ? "127.0.0.1" : entity.getHost();
        String infoUrl = "http://" + host + ":" + entity.getPort() + "/info";

        ResponseEntity<Map> response = restTemplate.getForEntity(infoUrl, Map.class);
        return response.getBody();
    }

    public Map<String, Object> callHealthCheck(String serverId) {
        ModelServerEntity entity = serverStore.get(serverId);
        if (entity == null) {
            throw new IllegalArgumentException("Server not found: " + serverId);
        }

        String host = "0.0.0.0".equals(entity.getHost()) ? "127.0.0.1" : entity.getHost();
        String healthUrl = "http://" + host + ":" + entity.getPort() + "/health";

        ResponseEntity<Map> response = restTemplate.getForEntity(healthUrl, Map.class);
        return response.getBody();
    }

    public Map<String, Object> predict(String serverId, Map<String, Object> inputData) {
        ModelServerEntity entity = serverStore.get(serverId);
        if (entity == null) {
            throw new IllegalArgumentException("Server not found: " + serverId);
        }

        if (entity.getStatus() != ModelServerEntity.Status.RUNNING) {
            throw new IllegalStateException("Server is not running: " + entity.getStatus());
        }

        String host = "0.0.0.0".equals(entity.getHost()) ? "127.0.0.1" : entity.getHost();
        String predictUrl = "http://" + host + ":" + entity.getPort() + "/predict";

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> requestEntity = new HttpEntity<>(inputData, headers);

        ResponseEntity<Map> response = restTemplate.postForEntity(predictUrl, requestEntity, Map.class);
        return response.getBody();
    }

    public Map<String, Object> reloadModel(String serverId, String modelPath, String framework) {
        ModelServerEntity entity = serverStore.get(serverId);
        if (entity == null) {
            throw new IllegalArgumentException("Server not found: " + serverId);
        }

        if (entity.getStatus() != ModelServerEntity.Status.RUNNING) {
            throw new IllegalStateException("Server is not running: " + entity.getStatus());
        }

        String host = "0.0.0.0".equals(entity.getHost()) ? "127.0.0.1" : entity.getHost();
        String reloadUrl = "http://" + host + ":" + entity.getPort() + "/reload";

        Map<String, Object> requestData = new HashMap<>();
        if (modelPath != null && !modelPath.isEmpty()) {
            requestData.put("model_path", modelPath);
        }
        if (framework != null && !framework.isEmpty()) {
            requestData.put("framework", framework);
        }

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> requestEntity = new HttpEntity<>(requestData, headers);

        ResponseEntity<Map> response = restTemplate.postForEntity(reloadUrl, requestEntity, Map.class);

        Map<String, Object> result = response.getBody();
        if (result != null && Boolean.TRUE.equals(result.get("success"))) {
            if (modelPath != null && !modelPath.isEmpty()) {
                entity.setModelPath(modelPath);
            }
            if (result.get("framework") != null) {
                entity.setFramework(result.get("framework").toString());
            }
            if (result.get("device") != null) {
                entity.setDevice(result.get("device").toString());
            }
            entity.onUpdate();

            addServerLog(serverId, "Model reloaded: " + entity.getModelPath());
        }

        return result;
    }
}
