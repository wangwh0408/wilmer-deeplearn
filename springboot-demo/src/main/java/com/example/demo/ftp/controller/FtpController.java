package com.example.demo.ftp.controller;

import com.example.demo.common.Result;
import com.example.demo.ftp.config.FtpProperties;
import com.example.demo.ftp.dto.FtpConnectionRequest;
import com.example.demo.ftp.dto.FtpDirectoryListing;
import com.example.demo.ftp.dto.FtpFileInfo;
import com.example.demo.ftp.dto.FtpFolderUploadResult;
import com.example.demo.ftp.service.FtpService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import javax.servlet.http.HttpServletResponse;
import javax.validation.constraints.NotBlank;
import java.util.*;

@Slf4j
@RestController
@RequestMapping("/api/ftp")
@Validated
public class FtpController {

    @Autowired
    private FtpService ftpService;

    @Autowired
    private FtpProperties properties;

    @GetMapping("/info")
    public Result<Map<String, Object>> getApiInfo() {
        Map<String, Object> info = new LinkedHashMap<>();
        info.put("name", "FTP Service API");
        info.put("version", "1.0.0");
        info.put("description", "High availability FTP service with connection pooling");
        
        Map<String, Object> endpoints = new LinkedHashMap<>();
        endpoints.put("POST /api/ftp/test", "Test FTP connection");
        endpoints.put("GET /api/ftp/list", "List files in directory");
        endpoints.put("GET /api/ftp/file-info", "Get file information");
        endpoints.put("POST /api/ftp/upload", "Upload file");
        endpoints.put("POST /api/ftp/upload-folder", "Upload folder with multiple files");
        endpoints.put("GET /api/ftp/download", "Download file");
        endpoints.put("POST /api/ftp/create-dir", "Create directory");
        endpoints.put("DELETE /api/ftp/delete", "Delete file or directory");
        endpoints.put("POST /api/ftp/rename", "Rename file or directory");
        endpoints.put("POST /api/ftp/copy", "Copy file");
        endpoints.put("GET /api/ftp/pool-stats", "Get connection pool statistics");
        
        info.put("endpoints", endpoints);
        info.put("poolConfig", properties.getPool());
        info.put("retryConfig", properties.getRetry());
        
        return Result.success(info);
    }

    @PostMapping("/test")
    public Result<Map<String, Object>> testConnection(@RequestBody FtpConnectionRequest request) {
        log.info("[FtpController] Testing connection to {}:{}", request.getHost(), request.getPort());
        
        boolean success = ftpService.testConnection(request);
        
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("connected", success);
        result.put("host", request.getHost());
        result.put("port", request.getPort());
        result.put("username", request.getUsername());
        
        if (success) {
            log.info("[FtpController] Connection test successful");
            return Result.success(result);
        } else {
            log.warn("[FtpController] Connection test failed");
            return Result.error(503, "Connection failed");
        }
    }

    @GetMapping("/list")
    public Result<FtpDirectoryListing> listFiles(
            @RequestParam(required = false) String serverName,
            @RequestParam(required = false) String host,
            @RequestParam(required = false, defaultValue = "21") int port,
            @RequestParam(required = false) String username,
            @RequestParam(required = false) String password,
            @RequestParam(required = false, defaultValue = "/") String path,
            @RequestParam(required = false) String rootPath) {
        
        FtpConnectionRequest request = buildConnectionRequest(serverName, host, port, username, password, rootPath);
        
        log.debug("[FtpController] Listing files in {} on {}:{}", path, request.getHost(), request.getPort());
        
        FtpDirectoryListing listing = ftpService.listFiles(request, path);
        
        log.debug("[FtpController] Found {} files/directories", listing.getTotalFiles());
        return Result.success(listing);
    }

    @GetMapping("/file-info")
    public Result<FtpFileInfo> getFileInfo(
            @RequestParam(required = false) String serverName,
            @RequestParam(required = false) String host,
            @RequestParam(required = false, defaultValue = "21") int port,
            @RequestParam(required = false) String username,
            @RequestParam(required = false) String password,
            @RequestParam @NotBlank(message = "Path cannot be blank") String path,
            @RequestParam(required = false) String rootPath) {
        
        FtpConnectionRequest request = buildConnectionRequest(serverName, host, port, username, password, rootPath);
        
        log.debug("[FtpController] Getting file info for {} on {}:{}", path, request.getHost(), request.getPort());
        
        FtpFileInfo fileInfo = ftpService.getFileInfo(request, path);
        
        return Result.success(fileInfo);
    }

    @PostMapping("/upload")
    public Result<Map<String, Object>> uploadFile(
            @RequestParam(required = false) String serverName,
            @RequestParam(required = false) String host,
            @RequestParam(required = false, defaultValue = "21") int port,
            @RequestParam(required = false) String username,
            @RequestParam(required = false) String password,
            @RequestParam(required = false) String rootPath,
            @RequestParam(required = false, defaultValue = "/") String targetPath,
            @RequestParam(required = false, defaultValue = "false") boolean overwrite,
            @RequestParam("file") MultipartFile file) {
        
        FtpConnectionRequest request = buildConnectionRequest(serverName, host, port, username, password, rootPath);
        
        log.info("[FtpController] Uploading file '{}' to {} on {}:{}", 
            file.getOriginalFilename(), targetPath, request.getHost(), request.getPort());
        
        if (file.isEmpty()) {
            return Result.error(400, "File is empty");
        }
        
        boolean success = ftpService.uploadFile(request, file, targetPath, overwrite);
        
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("success", success);
        result.put("fileName", file.getOriginalFilename());
        result.put("fileSize", file.getSize());
        result.put("targetPath", targetPath);
        result.put("overwrite", overwrite);
        
        if (success) {
            log.info("[FtpController] File uploaded successfully: {}", file.getOriginalFilename());
            return Result.success(result);
        } else {
            return Result.error(500, "Upload failed");
        }
    }

    @GetMapping("/download")
    public void downloadFile(
            HttpServletResponse response,
            @RequestParam(required = false) String serverName,
            @RequestParam(required = false) String host,
            @RequestParam(required = false, defaultValue = "21") int port,
            @RequestParam(required = false) String username,
            @RequestParam(required = false) String password,
            @RequestParam(required = false) String rootPath,
            @RequestParam @NotBlank(message = "Path cannot be blank") String path) {
        
        FtpConnectionRequest request = buildConnectionRequest(serverName, host, port, username, password, rootPath);
        
        log.info("[FtpController] Downloading file: {} from {}:{}", path, request.getHost(), request.getPort());
        
        ftpService.downloadFile(request, path, response);
    }

    @PostMapping("/create-dir")
    public Result<Map<String, Object>> createDirectory(
            @RequestParam(required = false) String serverName,
            @RequestParam(required = false) String host,
            @RequestParam(required = false, defaultValue = "21") int port,
            @RequestParam(required = false) String username,
            @RequestParam(required = false) String password,
            @RequestParam(required = false) String rootPath,
            @RequestParam @NotBlank(message = "Path cannot be blank") String path) {
        
        FtpConnectionRequest request = buildConnectionRequest(serverName, host, port, username, password, rootPath);
        
        log.info("[FtpController] Creating directory: {} on {}:{}", path, request.getHost(), request.getPort());
        
        boolean success = ftpService.createDirectory(request, path);
        
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("success", success);
        result.put("path", path);
        
        if (success) {
            log.info("[FtpController] Directory created successfully: {}", path);
            return Result.success(result);
        } else {
            return Result.error(500, "Create directory failed");
        }
    }

    @DeleteMapping("/delete")
    public Result<Map<String, Object>> deleteFile(
            @RequestParam(required = false) String serverName,
            @RequestParam(required = false) String host,
            @RequestParam(required = false, defaultValue = "21") int port,
            @RequestParam(required = false) String username,
            @RequestParam(required = false) String password,
            @RequestParam(required = false) String rootPath,
            @RequestParam @NotBlank(message = "Path cannot be blank") String path) {
        
        FtpConnectionRequest request = buildConnectionRequest(serverName, host, port, username, password, rootPath);
        
        log.info("[FtpController] Deleting: {} on {}:{}", path, request.getHost(), request.getPort());
        
        boolean success = ftpService.deleteFile(request, path);
        
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("success", success);
        result.put("path", path);
        
        if (success) {
            log.info("[FtpController] Deleted successfully: {}", path);
            return Result.success(result);
        } else {
            return Result.error(500, "Delete failed");
        }
    }

    @PostMapping("/rename")
    public Result<Map<String, Object>> renameFile(
            @RequestParam(required = false) String serverName,
            @RequestParam(required = false) String host,
            @RequestParam(required = false, defaultValue = "21") int port,
            @RequestParam(required = false) String username,
            @RequestParam(required = false) String password,
            @RequestParam(required = false) String rootPath,
            @RequestParam @NotBlank(message = "Old path cannot be blank") String oldPath,
            @RequestParam @NotBlank(message = "New path cannot be blank") String newPath) {
        
        FtpConnectionRequest request = buildConnectionRequest(serverName, host, port, username, password, rootPath);
        
        log.info("[FtpController] Renaming {} to {} on {}:{}", 
            oldPath, newPath, request.getHost(), request.getPort());
        
        boolean success = ftpService.renameFile(request, oldPath, newPath);
        
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("success", success);
        result.put("oldPath", oldPath);
        result.put("newPath", newPath);
        
        if (success) {
            log.info("[FtpController] Renamed successfully");
            return Result.success(result);
        } else {
            return Result.error(500, "Rename failed");
        }
    }

    @PostMapping("/copy")
    public Result<Map<String, Object>> copyFile(
            @RequestParam(required = false) String serverName,
            @RequestParam(required = false) String host,
            @RequestParam(required = false, defaultValue = "21") int port,
            @RequestParam(required = false) String username,
            @RequestParam(required = false) String password,
            @RequestParam(required = false) String rootPath,
            @RequestParam @NotBlank(message = "Source path cannot be blank") String sourcePath,
            @RequestParam @NotBlank(message = "Target path cannot be blank") String targetPath) {
        
        FtpConnectionRequest request = buildConnectionRequest(serverName, host, port, username, password, rootPath);
        
        log.info("[FtpController] Copying {} to {} on {}:{}", 
            sourcePath, targetPath, request.getHost(), request.getPort());
        
        boolean success = ftpService.copyFile(request, sourcePath, targetPath);
        
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("success", success);
        result.put("sourcePath", sourcePath);
        result.put("targetPath", targetPath);
        
        if (success) {
            log.info("[FtpController] Copied successfully");
            return Result.success(result);
        } else {
            return Result.error(500, "Copy failed");
        }
    }

    @PostMapping("/upload-folder")
    public Result<FtpFolderUploadResult> uploadFolder(
            @RequestParam(required = false) String serverName,
            @RequestParam(required = false) String host,
            @RequestParam(required = false, defaultValue = "21") int port,
            @RequestParam(required = false) String username,
            @RequestParam(required = false) String password,
            @RequestParam(required = false) String rootPath,
            @RequestParam(required = false, defaultValue = "/") String targetBasePath,
            @RequestParam(required = false, defaultValue = "false") boolean overwrite,
            @RequestParam(value = "files", required = false) List<MultipartFile> files,
            @RequestParam(value = "relativePaths", required = false) List<String> relativePaths) {
        
        FtpConnectionRequest request = buildConnectionRequest(serverName, host, port, username, password, rootPath);
        
        log.info("[FtpController] Uploading folder to {} on {}:{}", 
            targetBasePath, request.getHost(), request.getPort());
        
        if (files == null || files.isEmpty()) {
            return Result.error(400, "No files provided");
        }
        
        if (files.size() > 0) {
            log.info("[FtpController] Uploading {} files for folder", files.size());
        }
        
        FtpFolderUploadResult result = ftpService.uploadFolder(
            request, files, relativePaths, targetBasePath, overwrite);
        
        if (result.isAllSuccess()) {
            log.info("[FtpController] Folder uploaded successfully: {} files", result.getUploadedFiles());
            return Result.success(result);
        } else {
            log.warn("[FtpController] Folder upload partially completed: {}/{} files successful", 
                result.getUploadedFiles(), result.getTotalFiles());
            return Result.success(result);
        }
    }

    @GetMapping("/pool-stats")
    public Result<Map<String, Object>> getPoolStats() {
        log.debug("[FtpController] Getting pool statistics");
        
        Map<String, Object> stats = ftpService.getPoolStats();
        
        return Result.success(stats);
    }

    private FtpConnectionRequest buildConnectionRequest(
            String serverName, String host, int port, 
            String username, String password, String rootPath) {
        
        if (serverName != null && properties.getServers().containsKey(serverName)) {
            FtpProperties.FtpServerConfig config = properties.getServers().get(serverName);
            return FtpConnectionRequest.builder()
                .serverName(serverName)
                .host(config.getHost())
                .port(config.getPort())
                .username(config.getUsername() != null ? config.getUsername() : "anonymous")
                .password(config.getPassword() != null ? config.getPassword() : "")
                .rootPath(config.getRootPath() != null ? config.getRootPath() : "/")
                .connectTimeout(config.getConnectTimeout())
                .dataTimeout(config.getDataTimeout())
                .passiveMode(config.isPassiveMode())
                .encoding(config.getEncoding())
                .build();
        }
        
        if (host == null || host.isEmpty()) {
            throw new IllegalArgumentException("Either serverName or host must be provided");
        }
        
        return FtpConnectionRequest.builder()
            .serverName(serverName)
            .host(host)
            .port(port)
            .username(username != null ? username : "anonymous")
            .password(password != null ? password : "")
            .rootPath(rootPath != null ? rootPath : "/")
            .build();
    }
}
