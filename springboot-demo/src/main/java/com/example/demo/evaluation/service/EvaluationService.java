package com.example.demo.evaluation.service;

import com.example.demo.evaluation.dto.EvaluationRequest;
import com.example.demo.evaluation.entity.EvaluationLogEntry;
import com.example.demo.evaluation.entity.EvaluationTask;
import com.example.demo.evaluation.manager.EvaluationTaskManager;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

@Slf4j
@Service
public class EvaluationService {

    @Autowired
    private EvaluationTaskManager taskManager;

    private String defaultWorkingDir;
    private String pythonExecutablePath;
    private String evaluationScriptPath;
    private String defaultOutputReportPath;
    
    private final ExecutorService executorService = Executors.newCachedThreadPool();
    private final Map<String, Process> runningProcesses = new ConcurrentHashMap<>();
    private final Map<String, Future<?>> runningFutures = new ConcurrentHashMap<>();
    private final ObjectMapper objectMapper = new ObjectMapper();

    @PostConstruct
    public void init() {
        String userDir = System.getProperty("user.dir");
        File projectDir = new File(userDir).getParentFile();
        if (projectDir == null) {
            projectDir = new File(userDir);
        }
        
        defaultWorkingDir = projectDir.getAbsolutePath();
        log.info("[EvaluationService] Default working directory set to: {}", defaultWorkingDir);
        
        findPythonExecutableAndScript();
    }
    
    private void findPythonExecutableAndScript() {
        String os = System.getProperty("os.name").toLowerCase();
        boolean isWindows = os.contains("win");
        
        String portablePythonPath = defaultWorkingDir + File.separator + "portable_env" + 
            File.separator + "python-portable" + File.separator + "python" + 
            File.separator + "python.exe";
        
        File portablePythonFile = new File(portablePythonPath);
        if (portablePythonFile.exists()) {
            pythonExecutablePath = portablePythonPath;
            log.info("[EvaluationService] Found portable Python: {}", pythonExecutablePath);
        } else {
            pythonExecutablePath = isWindows ? "python" : "python3";
            log.info("[EvaluationService] Using system Python: {}", pythonExecutablePath);
        }
        
        String workspaceDir = defaultWorkingDir + File.separator + "portable_env" + 
            File.separator + "python-portable" + File.separator + "workspace";
        
        evaluationScriptPath = workspaceDir + File.separator + "evaluate_navier_stokes_enhanced.py";
        File scriptFile = new File(evaluationScriptPath);
        if (!scriptFile.exists()) {
            evaluationScriptPath = "evaluate_navier_stokes_enhanced.py";
            log.warn("[EvaluationService] Script not found in workspace, using: {}", evaluationScriptPath);
        } else {
            log.info("[EvaluationService] Found evaluation script: {}", evaluationScriptPath);
        }
        
        defaultOutputReportPath = workspaceDir + File.separator + "evaluation_report_enhanced.json";
        log.info("[EvaluationService] Default output report path: {}", defaultOutputReportPath);
    }

    public String startEvaluation(EvaluationRequest request) {
        return startEvaluation(request, defaultWorkingDir);
    }

    public String startEvaluation(EvaluationRequest request, String workingDir) {
        String taskId = taskManager.createTask();
        EvaluationTask task = taskManager.getTask(taskId);

        task.addLog(EvaluationLogEntry.info(taskId, task.getNextSequence(), 
            "Starting model evaluation task..."));
        
        if (request.getModelPath() != null) {
            task.setModelPath(request.getModelPath());
            task.addLog(EvaluationLogEntry.info(taskId, task.getNextSequence(), 
                "Model path: " + request.getModelPath()));
        }
        if (request.getDatasetPath() != null) {
            task.setDatasetPath(request.getDatasetPath());
            task.addLog(EvaluationLogEntry.info(taskId, task.getNextSequence(), 
                "Dataset path: " + request.getDatasetPath()));
        }
        if (request.getProblems() != null) {
            task.setProblems(request.getProblems());
            task.addLog(EvaluationLogEntry.info(taskId, task.getNextSequence(), 
                "Problems: " + String.join(",", request.getProblems())));
        }
        if (request.getCategories() != null) {
            task.setCategories(request.getCategories());
            task.addLog(EvaluationLogEntry.info(taskId, task.getNextSequence(), 
                "Categories: " + String.join(",", request.getCategories())));
        }
        if (request.getMetrics() != null) {
            task.addLog(EvaluationLogEntry.info(taskId, task.getNextSequence(), 
                "Metrics: " + String.join(",", request.getMetrics())));
        }

        String actualWorkingDir = request.getWorkingDirectory() != null 
            ? request.getWorkingDirectory() 
            : workingDir;
        
        if (request.getOutputReportPath() != null) {
            task.setOutputReportPath(request.getOutputReportPath());
        } else {
            String taskReportPath = actualWorkingDir + File.separator + 
                "evaluation_report_" + taskId + ".json";
            task.setOutputReportPath(taskReportPath);
        }
        
        task.addLog(EvaluationLogEntry.info(taskId, task.getNextSequence(), 
            "Output report: " + task.getOutputReportPath()));
        task.addLog(EvaluationLogEntry.info(taskId, task.getNextSequence(), 
            "Working directory: " + actualWorkingDir));

        try {
            List<String> command = buildCommand(request, task);
            task.setCommand(String.join(" ", command));
            task.addLog(EvaluationLogEntry.info(taskId, task.getNextSequence(), 
                "Command: " + task.getCommand()));

            ProcessBuilder processBuilder = new ProcessBuilder(command);
            processBuilder.directory(new File(actualWorkingDir));
            processBuilder.redirectErrorStream(true);

            Map<String, String> env = processBuilder.environment();
            env.put("PYTHONUNBUFFERED", "1");
            env.put("PYTHONDONTWRITEBYTECODE", "1");
            
            task.addLog(EvaluationLogEntry.info(taskId, task.getNextSequence(), 
                "Environment: PYTHONUNBUFFERED=1 (no buffering)"));

            task.markRunning();

            Process process = processBuilder.start();
            runningProcesses.put(taskId, process);

            Future<?> future = executorService.submit(() -> {
                try {
                    runEvaluationProcess(taskId, process);
                } catch (Exception e) {
                    log.error("[EvaluationService] Error in evaluation process for task: {}", taskId, e);
                    EvaluationTask t = taskManager.getTask(taskId);
                    if (t != null) {
                        t.addLog(EvaluationLogEntry.error(t.getTaskId(), t.getNextSequence(), 
                            "Error: " + e.getMessage()));
                        t.markFailed(e.getMessage());
                    }
                } finally {
                    runningProcesses.remove(taskId);
                    runningFutures.remove(taskId);
                }
            });
            runningFutures.put(taskId, future);

            return taskId;

        } catch (Exception e) {
            log.error("[EvaluationService] Failed to start evaluation task: {}", taskId, e);
            task.addLog(EvaluationLogEntry.error(taskId, task.getNextSequence(), 
                "Failed to start evaluation: " + e.getMessage()));
            task.markFailed(e.getMessage());
            return taskId;
        }
    }

    private List<String> buildCommand(EvaluationRequest request, EvaluationTask task) {
        List<String> command = new ArrayList<>();
        
        command.add(pythonExecutablePath);
        command.add("-u");
        command.add(evaluationScriptPath);
        
        if (request.getModelPath() != null && !request.getModelPath().isEmpty()) {
            command.add("--model_path");
            command.add(request.getModelPath());
        }
        
        if (request.getDatasetPath() != null && !request.getDatasetPath().isEmpty()) {
            command.add("--data_root");
            command.add(request.getDatasetPath());
        }
        
        if (request.getProblems() != null && !request.getProblems().isEmpty()) {
            command.add("--problems");
            command.add(String.join(",", request.getProblems()));
        }
        
        if (request.getCategories() != null && !request.getCategories().isEmpty()) {
            command.add("--categories");
            command.add(String.join(",", request.getCategories()));
        }
        
        Integer maxCasesPerCategory = request.getMaxCasesPerCategory() != null 
            ? request.getMaxCasesPerCategory() 
            : request.getMaxCases();
        if (maxCasesPerCategory != null && maxCasesPerCategory > 0) {
            command.add("--max_cases_per_category");
            command.add(maxCasesPerCategory.toString());
        }
        
        if (request.getNormalize() != null && !request.getNormalize()) {
            command.add("--no_normalize");
        }
        
        if (request.getDevice() != null && !request.getDevice().isEmpty()) {
            command.add("--device");
            command.add(request.getDevice());
        }
        
        if (request.getMetrics() != null && !request.getMetrics().isEmpty()) {
            command.add("--metrics");
            command.add(String.join(",", request.getMetrics()));
        }
        
        if (request.getLogLevel() != null && !request.getLogLevel().isEmpty()) {
            command.add("--log_level");
            command.add(request.getLogLevel());
        }
        
        if (request.getNoFileLog() != null && request.getNoFileLog()) {
            command.add("--no_file_log");
        }
        
        if (request.getPrintEvery() != null && request.getPrintEvery() > 0) {
            command.add("--print_every");
            command.add(request.getPrintEvery().toString());
        }
        
        if (request.getBatchSize() != null && request.getBatchSize() > 0) {
            command.add("--batch_size");
            command.add(request.getBatchSize().toString());
        }
        
        if (task.getOutputReportPath() != null && !task.getOutputReportPath().isEmpty()) {
            command.add("--output_report");
            command.add(task.getOutputReportPath());
        }
        
        return command;
    }

    private void runEvaluationProcess(String taskId, Process process) {
        EvaluationTask task = taskManager.getTask(taskId);
        if (task == null) {
            log.warn("[EvaluationService] Task not found: {}", taskId);
            return;
        }

        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
            
            String line;
            while ((line = reader.readLine()) != null) {
                synchronized (task) {
                    int seq = task.getNextSequence();
                    task.addLog(EvaluationLogEntry.pythonOutput(taskId, seq, line));
                }
                
                log.debug("[EvaluationProcess] {}: {}", taskId, line);
            }
        } catch (Exception e) {
            log.error("[EvaluationService] Error reading process output for task: {}", taskId, e);
            task.addLog(EvaluationLogEntry.error(taskId, task.getNextSequence(), 
                "Error reading output: " + e.getMessage()));
        }

        try {
            int exitCode = process.waitFor();
            task.setExitCode(exitCode);

            if (exitCode == 0) {
                parseFinalReport(task);
                task.markCompleted();
                synchronized (task) {
                    task.addLog(EvaluationLogEntry.info(taskId, task.getNextSequence(), 
                        "Evaluation completed successfully!"));
                }
                log.info("[EvaluationService] Evaluation task completed successfully: {}", taskId);
            } else {
                parseFinalReport(task);
                task.markFailed("Process exited with code: " + exitCode);
                synchronized (task) {
                    task.addLog(EvaluationLogEntry.error(taskId, task.getNextSequence(), 
                        "Evaluation failed with exit code: " + exitCode));
                }
                log.warn("[EvaluationService] Evaluation task failed with exit code {}: {}", exitCode, taskId);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            task.markFailed("Process interrupted: " + e.getMessage());
            log.warn("[EvaluationService] Evaluation task interrupted: {}", taskId);
        }
    }

    private void parseFinalReport(EvaluationTask task) {
        String reportPath = task.getOutputReportPath();
        if (reportPath == null || reportPath.isEmpty()) {
            reportPath = defaultOutputReportPath;
        }
        
        File reportFile = new File(reportPath);
        if (!reportFile.exists()) {
            log.warn("[EvaluationService] Report file not found: {}", reportPath);
            return;
        }

        try {
            JsonNode root = objectMapper.readTree(reportFile);
            EvaluationTask.EvaluationMetrics metrics = new EvaluationTask.EvaluationMetrics();
            
            if (root.has("model_path")) {
                metrics.setModelPath(root.path("model_path").asText());
            }
            if (root.has("data_root")) {
                metrics.setDataRoot(root.path("data_root").asText());
            }
            if (root.has("num_samples")) {
                metrics.setNumSamples(root.path("num_samples").asInt());
                metrics.setTotalSamples(root.path("num_samples").asInt());
            }
            if (root.has("average_loss")) {
                metrics.setAverageLoss(root.path("average_loss").asDouble());
            }
            if (root.has("evaluation_time")) {
                metrics.setEvaluationTime(root.path("evaluation_time").asText());
            }
            
            if (root.has("config")) {
                Map<String, Object> configMap = new HashMap<>();
                JsonNode configNode = root.path("config");
                Iterator<Map.Entry<String, JsonNode>> fields = configNode.fields();
                while (fields.hasNext()) {
                    Map.Entry<String, JsonNode> entry = fields.next();
                    configMap.put(entry.getKey(), jsonNodeToObject(entry.getValue()));
                }
                metrics.setConfig(configMap);
            }
            
            if (root.has("normalization_params")) {
                Map<String, Object> normMap = new HashMap<>();
                JsonNode normNode = root.path("normalization_params");
                Iterator<Map.Entry<String, JsonNode>> fields = normNode.fields();
                while (fields.hasNext()) {
                    Map.Entry<String, JsonNode> entry = fields.next();
                    normMap.put(entry.getKey(), jsonNodeToObject(entry.getValue()));
                }
                metrics.setNormalizationParams(normMap);
            }
            
            if (root.has("metrics")) {
                JsonNode metricsNode = root.path("metrics");
                
                if (metricsNode.has("u")) {
                    Map<String, Double> uMetrics = parseDynamicMetrics(metricsNode.path("u"));
                    metrics.setUMetrics(uMetrics);
                    log.info("[EvaluationService] Parsed U metrics: {}", uMetrics);
                }
                if (metricsNode.has("v")) {
                    Map<String, Double> vMetrics = parseDynamicMetrics(metricsNode.path("v"));
                    metrics.setVMetrics(vMetrics);
                    log.info("[EvaluationService] Parsed V metrics: {}", vMetrics);
                }
                if (metricsNode.has("combined")) {
                    Map<String, Double> combinedMetrics = parseDynamicMetrics(metricsNode.path("combined"));
                    metrics.setCombinedMetrics(combinedMetrics);
                    log.info("[EvaluationService] Parsed Combined metrics: {}", combinedMetrics);
                }
            }
            
            if (root.has("per_sample_stats")) {
                JsonNode perSampleStatsNode = root.path("per_sample_stats");
                
                if (perSampleStatsNode.has("u")) {
                    Map<String, Double> uStats = parseDynamicMetrics(perSampleStatsNode.path("u"));
                    metrics.setUPerSampleStats(uStats);
                }
                if (perSampleStatsNode.has("v")) {
                    Map<String, Double> vStats = parseDynamicMetrics(perSampleStatsNode.path("v"));
                    metrics.setVPerSampleStats(vStats);
                }
            }
            
            if (root.has("best_worst_samples")) {
                Map<String, Object> bestWorstMap = new HashMap<>();
                JsonNode bwNode = root.path("best_worst_samples");
                Iterator<Map.Entry<String, JsonNode>> fields = bwNode.fields();
                while (fields.hasNext()) {
                    Map.Entry<String, JsonNode> entry = fields.next();
                    bestWorstMap.put(entry.getKey(), jsonNodeToObject(entry.getValue()));
                }
                metrics.setBestWorstSamples(bestWorstMap);
            }
            
            task.setMetrics(metrics);
            log.info("[EvaluationService] Parsed metrics from report file: {}", reportPath);
            
        } catch (Exception e) {
            log.error("[EvaluationService] Failed to parse report file: {}", reportPath, e);
        }
    }
    
    private Map<String, Double> parseDynamicMetrics(JsonNode node) {
        Map<String, Double> metrics = new HashMap<>();
        if (node == null || node.isNull() || node.isMissingNode()) {
            return metrics;
        }
        
        Iterator<Map.Entry<String, JsonNode>> fields = node.fields();
        while (fields.hasNext()) {
            Map.Entry<String, JsonNode> entry = fields.next();
            if (entry.getValue().isNumber()) {
                metrics.put(entry.getKey(), entry.getValue().asDouble());
            }
        }
        return metrics;
    }
    
    private Object jsonNodeToObject(JsonNode node) {
        if (node == null || node.isNull() || node.isMissingNode()) {
            return null;
        }
        if (node.isBoolean()) {
            return node.asBoolean();
        }
        if (node.isInt()) {
            return node.asInt();
        }
        if (node.isLong()) {
            return node.asLong();
        }
        if (node.isDouble()) {
            return node.asDouble();
        }
        if (node.isTextual()) {
            return node.asText();
        }
        if (node.isArray()) {
            List<Object> list = new ArrayList<>();
            for (JsonNode element : node) {
                list.add(jsonNodeToObject(element));
            }
            return list;
        }
        if (node.isObject()) {
            Map<String, Object> map = new HashMap<>();
            Iterator<Map.Entry<String, JsonNode>> fields = node.fields();
            while (fields.hasNext()) {
                Map.Entry<String, JsonNode> entry = fields.next();
                map.put(entry.getKey(), jsonNodeToObject(entry.getValue()));
            }
            return map;
        }
        return node.asText();
    }

    public EvaluationTask getTaskStatus(String taskId) {
        return taskManager.getTask(taskId);
    }

    public List<EvaluationLogEntry> getLogs(String taskId, Integer since) {
        EvaluationTask task = taskManager.getTask(taskId);
        if (task == null) {
            return new ArrayList<>();
        }
        if (since != null && since >= 0) {
            return task.getLogsSince(since);
        }
        return task.getLogs();
    }

    public List<EvaluationLogEntry> getNewLogs(String taskId) {
        EvaluationTask task = taskManager.getTask(taskId);
        if (task == null) {
            return new ArrayList<>();
        }
        return task.getNewLogs();
    }

    public boolean cancelEvaluation(String taskId) {
        EvaluationTask task = taskManager.getTask(taskId);
        if (task == null) {
            return false;
        }

        if (!task.isRunning()) {
            log.warn("[EvaluationService] Task is not running: {}", taskId);
            return false;
        }

        Process process = runningProcesses.get(taskId);
        if (process != null && process.isAlive()) {
            process.destroy();
            log.info("[EvaluationService] Destroyed evaluation process: {}", taskId);
        }

        Future<?> future = runningFutures.get(taskId);
        if (future != null && !future.isDone()) {
            future.cancel(true);
        }

        task.markCancelled();
        task.addLog(EvaluationLogEntry.warning(taskId, task.getNextSequence(), 
            "Evaluation cancelled by user"));
        
        return true;
    }

    public List<EvaluationTask> getAllTasks() {
        return taskManager.getAllTasks();
    }
    
    public String getPythonExecutablePath() {
        return pythonExecutablePath;
    }
    
    public String getEvaluationScriptPath() {
        return evaluationScriptPath;
    }
    
    public String getDefaultOutputReportPath() {
        return defaultOutputReportPath;
    }
}
