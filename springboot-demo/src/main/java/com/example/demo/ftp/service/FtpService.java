package com.example.demo.ftp.service;

import com.example.demo.ftp.config.FtpProperties;
import com.example.demo.ftp.dto.FtpConnectionRequest;
import com.example.demo.ftp.dto.FtpDirectoryListing;
import com.example.demo.ftp.dto.FtpFileInfo;
import com.example.demo.ftp.dto.FtpFolderUploadResult;
import com.example.demo.ftp.exception.*;
import com.example.demo.ftp.pool.FtpClientPool;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.net.ftp.FTPClient;
import org.apache.commons.net.ftp.FTPFile;
import org.apache.commons.net.ftp.FTPReply;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import javax.servlet.http.HttpServletResponse;
import java.io.*;
import java.net.SocketException;
import java.net.SocketTimeoutException;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.Callable;

@Slf4j
@Service
public class FtpService {

    @Autowired
    private FtpClientPool ftpClientPool;

    @Autowired
    private FtpProperties properties;

    private static final DateTimeFormatter DATE_FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    public boolean testConnection(FtpConnectionRequest request) {
        return executeWithRetry(() -> {
            FTPClient client = null;
            try {
                client = ftpClientPool.borrowObject(request);
                return client != null && client.isConnected();
            } catch (Exception e) {
                throw new FtpConnectionException("Connection test failed: " + e.getMessage(), e);
            } finally {
                if (client != null) {
                    ftpClientPool.returnObject(request, client);
                }
            }
        }, "test connection");
    }

    public FtpDirectoryListing listFiles(FtpConnectionRequest request, String path) {
        return executeWithRetry(() -> {
            FTPClient client = null;
            try {
                client = ftpClientPool.borrowObject(request);
                
                String currentPath = path == null || path.isEmpty() ? "/" : normalizePath(path);
                boolean changeSuccess = client.changeWorkingDirectory(currentPath);
                
                if (!changeSuccess) {
                    int replyCode = client.getReplyCode();
                    String replyString = client.getReplyString();
                    log.warn("[FtpService] Failed to change working directory to: {}, reply code: {}, reply: {}", 
                        currentPath, replyCode, replyString);
                    throw new FtpFileNotFoundException(currentPath, String.valueOf(replyCode));
                }
                
                String workingDir = client.printWorkingDirectory();
                log.info("[FtpService] Listing files in: {}", workingDir);
                
                FTPFile[] ftpFiles = null;
                
                ftpFiles = client.listFiles();
                if (ftpFiles == null || ftpFiles.length == 0) {
                    log.warn("[FtpService] listFiles() returned empty, trying listFiles(workingDir)...");
                    ftpFiles = client.listFiles(workingDir);
                }
                
                if (ftpFiles == null || ftpFiles.length == 0) {
                    log.warn("[FtpService] listFiles(workingDir) returned empty, trying listFiles(\".\")...");
                    ftpFiles = client.listFiles(".");
                }
                
                if (ftpFiles == null) {
                    log.warn("[FtpService] All listFiles attempts returned null, using empty array");
                    ftpFiles = new FTPFile[0];
                }
                
                log.info("[FtpService] Found {} files/directories in {}", ftpFiles.length, workingDir);
                
                List<FtpFileInfo> fileInfoList = new ArrayList<>();
                int dirCount = 0;
                int fileCount = 0;
                
                if (ftpFiles != null) {
                    for (FTPFile ftpFile : ftpFiles) {
                        if (ftpFile == null) continue;
                        
                        String name = ftpFile.getName();
                        if (name == null || name.isEmpty() || name.equals(".") || name.equals("..")) {
                            continue;
                        }
                        
                        FtpFileInfo info = convertToFileInfo(ftpFile, workingDir);
                        fileInfoList.add(info);
                        
                        log.debug("[FtpService] File: {}, isDirectory: {}, isFile: {}", 
                            name, info.isDirectory(), info.isFile());
                        
                        if (info.isDirectory()) {
                            dirCount++;
                        } else if (info.isFile()) {
                            fileCount++;
                        }
                    }
                }
                
                fileInfoList.sort((a, b) -> {
                    if (a.isDirectory() != b.isDirectory()) {
                        return a.isDirectory() ? -1 : 1;
                    }
                    return a.getName().compareToIgnoreCase(b.getName());
                });
                
                String parentPath = getParentPath(workingDir);
                List<String> pathParts = parsePathParts(workingDir);
                
                log.info("[FtpService] Returning {} files/directories ({} dirs, {} files) from {}", 
                    fileInfoList.size(), dirCount, fileCount, workingDir);
                
                return FtpDirectoryListing.builder()
                    .currentPath(workingDir)
                    .parentPath(parentPath)
                    .isRoot("/".equals(workingDir))
                    .pathParts(pathParts)
                    .files(fileInfoList)
                    .totalFiles(fileInfoList.size())
                    .directoryCount(dirCount)
                    .fileCount(fileCount)
                    .build();
                    
            } catch (FtpException e) {
                throw e;
            } catch (SocketTimeoutException e) {
                throw new FtpTimeoutException("list files", path, properties.getDataTimeout());
            } catch (IOException e) {
                log.error("[FtpService] IO error listing files: {}", e.getMessage(), e);
                throw new FtpException("IO error while listing files: " + e.getMessage(), e);
            } finally {
                if (client != null) {
                    ftpClientPool.returnObject(request, client);
                }
            }
        }, "list files at " + path);
    }

    public FtpFileInfo getFileInfo(FtpConnectionRequest request, String path) {
        return executeWithRetry(() -> {
            FTPClient client = null;
            try {
                client = ftpClientPool.borrowObject(request);
                
                String normalizedPath = normalizePath(path);
                String parentDir = getParentPath(normalizedPath);
                String fileName = getFileName(normalizedPath);
                
                if ("/".equals(normalizedPath)) {
                    return FtpFileInfo.builder()
                        .name("/")
                        .path("/")
                        .absolutePath("/")
                        .parentPath(null)
                        .isDirectory(true)
                        .isFile(false)
                        .size(0)
                        .build();
                }
                
                boolean changeSuccess = client.changeWorkingDirectory(parentDir);
                if (!changeSuccess) {
                    throw new FtpFileNotFoundException(parentDir);
                }
                
                FTPFile[] ftpFiles = client.listFiles(fileName);
                if (ftpFiles == null || ftpFiles.length == 0) {
                    throw new FtpFileNotFoundException(normalizedPath);
                }
                
                FTPFile ftpFile = ftpFiles[0];
                return convertToFileInfo(ftpFile, parentDir);
                    
            } catch (FtpException e) {
                throw e;
            } catch (IOException e) {
                throw new FtpException("IO error while getting file info: " + e.getMessage(), e);
            } finally {
                if (client != null) {
                    ftpClientPool.returnObject(request, client);
                }
            }
        }, "get file info for " + path);
    }

    public boolean uploadFile(FtpConnectionRequest request, MultipartFile file, String targetPath) {
        return uploadFile(request, file, targetPath, false);
    }

    public boolean uploadFile(FtpConnectionRequest request, MultipartFile file, String targetPath, boolean overwrite) {
        return executeWithRetry(() -> {
            FTPClient client = null;
            String targetDir = normalizePath(getParentPath(targetPath));
            String fileName = getFileName(targetPath);
            
            if (fileName == null || fileName.isEmpty()) {
                fileName = file.getOriginalFilename();
                if (fileName == null || fileName.isEmpty()) {
                    throw new IllegalArgumentException("File name cannot be empty");
                }
            }
            
            try {
                client = ftpClientPool.borrowObject(request);
                
                ensureDirectoryExists(client, targetDir);
                
                client.changeWorkingDirectory(targetDir);
                
                if (!overwrite) {
                    FTPFile[] existing = client.listFiles(fileName);
                    if (existing != null && existing.length > 0) {
                        throw new FtpPermissionException("File already exists", targetDir + "/" + fileName);
                    }
                }
                
                log.info("[FtpService] Uploading file {} to {}/{}", file.getOriginalFilename(), targetDir, fileName);
                
                try (InputStream inputStream = file.getInputStream()) {
                    boolean success = client.storeFile(fileName, inputStream);
                    
                    if (!success) {
                        int replyCode = client.getReplyCode();
                        String replyString = client.getReplyString();
                        log.error("[FtpService] Upload failed with reply code {}: {}", replyCode, replyString);
                        
                        if (replyCode == 552) {
                            throw new FtpException(507, "Storage exceeded: " + replyString);
                        } else if (replyCode == 553) {
                            throw new FtpPermissionException("upload", targetDir + "/" + fileName, String.valueOf(replyCode));
                        } else {
                            throw new FtpException(500, "Upload failed: " + replyString, String.valueOf(replyCode));
                        }
                    }
                    
                    log.info("[FtpService] Successfully uploaded file: {}/{}", targetDir, fileName);
                    return true;
                }
                    
            } catch (FtpException e) {
                throw e;
            } catch (SocketTimeoutException e) {
                throw new FtpTimeoutException("upload file", targetPath, properties.getDataTimeout());
            } catch (IOException e) {
                throw new FtpException("IO error while uploading file: " + e.getMessage(), e);
            } finally {
                if (client != null) {
                    ftpClientPool.returnObject(request, client);
                }
            }
        }, "upload file to " + targetPath);
    }

    public boolean uploadFile(FtpConnectionRequest request, File file, String targetPath, boolean overwrite) {
        return executeWithRetry(() -> {
            FTPClient client = null;
            String targetDir = normalizePath(getParentPath(targetPath));
            String fileName = getFileName(targetPath);
            
            if (fileName == null || fileName.isEmpty()) {
                fileName = file.getName();
                if (fileName == null || fileName.isEmpty()) {
                    throw new IllegalArgumentException("File name cannot be empty");
                }
            }
            
            try {
                client = ftpClientPool.borrowObject(request);
                
                ensureDirectoryExists(client, targetDir);
                
                client.changeWorkingDirectory(targetDir);
                
                if (!overwrite) {
                    FTPFile[] existing = client.listFiles(fileName);
                    if (existing != null && existing.length > 0) {
                        throw new FtpPermissionException("File already exists", targetDir + "/" + fileName);
                    }
                }
                
                log.info("[FtpService] Uploading file {} to {}/{}", file.getName(), targetDir, fileName);
                
                try (InputStream inputStream = new FileInputStream(file)) {
                    boolean success = client.storeFile(fileName, inputStream);
                    
                    if (!success) {
                        int replyCode = client.getReplyCode();
                        throw new FtpException(500, "Upload failed with reply code: " + replyCode, String.valueOf(replyCode));
                    }
                    
                    log.info("[FtpService] Successfully uploaded file: {}/{}", targetDir, fileName);
                    return true;
                }
                    
            } catch (FtpException e) {
                throw e;
            } catch (IOException e) {
                throw new FtpException("IO error while uploading file: " + e.getMessage(), e);
            } finally {
                if (client != null) {
                    ftpClientPool.returnObject(request, client);
                }
            }
        }, "upload file to " + targetPath);
    }

    public void downloadFile(FtpConnectionRequest request, String filePath, HttpServletResponse response) {
        executeWithRetry(() -> {
            FTPClient client = null;
            String normalizedPath = normalizePath(filePath);
            String fileName = getFileName(normalizedPath);
            String parentDir = getParentPath(normalizedPath);
            
            try {
                client = ftpClientPool.borrowObject(request);
                
                client.changeWorkingDirectory(parentDir);
                
                FTPFile[] ftpFiles = client.listFiles(fileName);
                if (ftpFiles == null || ftpFiles.length == 0) {
                    throw new FtpFileNotFoundException(normalizedPath);
                }
                
                FTPFile ftpFile = ftpFiles[0];
                if (ftpFile.isDirectory()) {
                    throw new IllegalArgumentException("Cannot download a directory");
                }
                
                log.info("[FtpService] Downloading file: {}", normalizedPath);
                
                String encodedFileName = URLEncoder.encode(fileName, StandardCharsets.UTF_8.name())
                    .replaceAll("\\+", "%20");
                
                response.setContentType("application/octet-stream");
                response.setHeader("Content-Disposition", "attachment; filename*=UTF-8''" + encodedFileName);
                response.setHeader("Content-Length", String.valueOf(ftpFile.getSize()));
                
                try (OutputStream outputStream = response.getOutputStream()) {
                    boolean success = client.retrieveFile(fileName, outputStream);
                    
                    if (!success) {
                        int replyCode = client.getReplyCode();
                        throw new FtpException(500, "Download failed with reply code: " + replyCode, String.valueOf(replyCode));
                    }
                    
                    log.info("[FtpService] Successfully downloaded file: {}", normalizedPath);
                }
                
                return null;
                    
            } catch (FtpException e) {
                throw e;
            } catch (SocketTimeoutException e) {
                throw new FtpTimeoutException("download file", filePath, properties.getDataTimeout());
            } catch (IOException e) {
                throw new FtpException("IO error while downloading file: " + e.getMessage(), e);
            } finally {
                if (client != null) {
                    ftpClientPool.returnObject(request, client);
                }
            }
        }, "download file " + filePath);
    }

    public byte[] downloadFileAsBytes(FtpConnectionRequest request, String filePath) {
        return executeWithRetry(() -> {
            FTPClient client = null;
            String normalizedPath = normalizePath(filePath);
            String fileName = getFileName(normalizedPath);
            String parentDir = getParentPath(normalizedPath);
            
            try {
                client = ftpClientPool.borrowObject(request);
                
                client.changeWorkingDirectory(parentDir);
                
                FTPFile[] ftpFiles = client.listFiles(fileName);
                if (ftpFiles == null || ftpFiles.length == 0) {
                    throw new FtpFileNotFoundException(normalizedPath);
                }
                
                FTPFile ftpFile = ftpFiles[0];
                if (ftpFile.isDirectory()) {
                    throw new IllegalArgumentException("Cannot download a directory");
                }
                
                log.info("[FtpService] Downloading file as bytes: {}", normalizedPath);
                
                try (ByteArrayOutputStream outputStream = new ByteArrayOutputStream()) {
                    boolean success = client.retrieveFile(fileName, outputStream);
                    
                    if (!success) {
                        int replyCode = client.getReplyCode();
                        throw new FtpException(500, "Download failed with reply code: " + replyCode, String.valueOf(replyCode));
                    }
                    
                    byte[] content = outputStream.toByteArray();
                    log.info("[FtpService] Successfully downloaded {} bytes", content.length);
                    return content;
                }
                    
            } catch (FtpException e) {
                throw e;
            } catch (IOException e) {
                throw new FtpException("IO error while downloading file: " + e.getMessage(), e);
            } finally {
                if (client != null) {
                    ftpClientPool.returnObject(request, client);
                }
            }
        }, "download file " + filePath);
    }

    public boolean createDirectory(FtpConnectionRequest request, String path) {
        return executeWithRetry(() -> {
            FTPClient client = null;
            String normalizedPath = normalizePath(path);
            
            try {
                client = ftpClientPool.borrowObject(request);
                
                log.info("[FtpService] Creating directory: {}", normalizedPath);
                
                String currentDir = client.printWorkingDirectory();
                log.debug("[FtpService] Current working directory before operation: {}", currentDir);
                
                ensureDirectoryExists(client, normalizedPath);
                
                String newDir = client.printWorkingDirectory();
                log.debug("[FtpService] Current working directory after operation: {}", newDir);
                
                log.info("[FtpService] Successfully created directory: {}", normalizedPath);
                return true;
                    
            } catch (FtpException e) {
                log.error("[FtpService] FtpException creating directory {}: {}", normalizedPath, e.getMessage());
                throw e;
            } catch (IOException e) {
                log.error("[FtpService] IOException creating directory {}: {}", normalizedPath, e.getMessage());
                throw new FtpException("IO error while creating directory: " + e.getMessage(), e);
            } finally {
                if (client != null) {
                    ftpClientPool.returnObject(request, client);
                }
            }
        }, "create directory " + path);
    }

    public boolean deleteFile(FtpConnectionRequest request, String path) {
        return executeWithRetry(() -> {
            FTPClient client = null;
            String normalizedPath = normalizePath(path);
            String parentDir = getParentPath(normalizedPath);
            String fileName = getFileName(normalizedPath);
            
            try {
                client = ftpClientPool.borrowObject(request);
                
                client.changeWorkingDirectory(parentDir);
                
                FTPFile[] ftpFiles = client.listFiles(fileName);
                if (ftpFiles == null || ftpFiles.length == 0) {
                    throw new FtpFileNotFoundException(normalizedPath);
                }
                
                FTPFile ftpFile = ftpFiles[0];
                boolean success;
                
                if (ftpFile.isDirectory()) {
                    success = deleteDirectoryRecursive(client, parentDir, fileName);
                } else {
                    success = client.deleteFile(fileName);
                }
                
                if (!success) {
                    int replyCode = client.getReplyCode();
                    if (replyCode == 550) {
                        throw new FtpPermissionException("delete", normalizedPath, String.valueOf(replyCode));
                    }
                    throw new FtpException(500, "Delete failed with reply code: " + replyCode, String.valueOf(replyCode));
                }
                
                log.info("[FtpService] Successfully deleted: {}", normalizedPath);
                return true;
                    
            } catch (FtpException e) {
                throw e;
            } catch (IOException e) {
                throw new FtpException("IO error while deleting: " + e.getMessage(), e);
            } finally {
                if (client != null) {
                    ftpClientPool.returnObject(request, client);
                }
            }
        }, "delete " + path);
    }

    public boolean renameFile(FtpConnectionRequest request, String oldPath, String newPath) {
        return executeWithRetry(() -> {
            FTPClient client = null;
            String normalizedOldPath = normalizePath(oldPath);
            String normalizedNewPath = normalizePath(newPath);
            
            try {
                client = ftpClientPool.borrowObject(request);
                
                String oldParent = getParentPath(normalizedOldPath);
                String oldName = getFileName(normalizedOldPath);
                String newParent = getParentPath(normalizedNewPath);
                String newName = getFileName(normalizedNewPath);
                
                client.changeWorkingDirectory(oldParent);
                
                FTPFile[] existing = client.listFiles(oldName);
                if (existing == null || existing.length == 0) {
                    throw new FtpFileNotFoundException(normalizedOldPath);
                }
                
                ensureDirectoryExists(client, newParent);
                
                String fromPath = normalizedOldPath;
                String toPath = normalizedNewPath;
                
                boolean success = client.rename(fromPath, toPath);
                
                if (!success) {
                    int replyCode = client.getReplyCode();
                    if (replyCode == 553) {
                        throw new FtpPermissionException("rename", normalizedNewPath, String.valueOf(replyCode));
                    }
                    throw new FtpException(500, "Rename failed with reply code: " + replyCode, String.valueOf(replyCode));
                }
                
                log.info("[FtpService] Successfully renamed {} to {}", normalizedOldPath, normalizedNewPath);
                return true;
                    
            } catch (FtpException e) {
                throw e;
            } catch (IOException e) {
                throw new FtpException("IO error while renaming: " + e.getMessage(), e);
            } finally {
                if (client != null) {
                    ftpClientPool.returnObject(request, client);
                }
            }
        }, "rename " + oldPath + " to " + newPath);
    }

    public boolean copyFile(FtpConnectionRequest request, String sourcePath, String targetPath) {
        return executeWithRetry(() -> {
            FTPClient client = null;
            String normalizedSource = normalizePath(sourcePath);
            String normalizedTarget = normalizePath(targetPath);
            
            try {
                client = ftpClientPool.borrowObject(request);
                
                String sourceParent = getParentPath(normalizedSource);
                String sourceName = getFileName(normalizedSource);
                String targetParent = getParentPath(normalizedTarget);
                String targetName = getFileName(normalizedTarget);
                
                client.changeWorkingDirectory(sourceParent);
                
                FTPFile[] sourceFiles = client.listFiles(sourceName);
                if (sourceFiles == null || sourceFiles.length == 0) {
                    throw new FtpFileNotFoundException(normalizedSource);
                }
                
                FTPFile sourceFile = sourceFiles[0];
                
                if (sourceFile.isDirectory()) {
                    throw new IllegalArgumentException("Copying directories is not supported");
                }
                
                ensureDirectoryExists(client, targetParent);
                
                log.info("[FtpService] Copying {} to {}", normalizedSource, normalizedTarget);
                
                try (ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
                    boolean retrieveSuccess = client.retrieveFile(sourceName, baos);
                    if (!retrieveSuccess) {
                        throw new FtpException("Failed to read source file");
                    }
                    
                    client.changeWorkingDirectory(targetParent);
                    
                    try (ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray())) {
                        boolean storeSuccess = client.storeFile(targetName, bais);
                        if (!storeSuccess) {
                            throw new FtpException("Failed to write target file");
                        }
                    }
                }
                
                log.info("[FtpService] Successfully copied {} to {}", normalizedSource, normalizedTarget);
                return true;
                    
            } catch (FtpException e) {
                throw e;
            } catch (IOException e) {
                throw new FtpException("IO error while copying: " + e.getMessage(), e);
            } finally {
                if (client != null) {
                    ftpClientPool.returnObject(request, client);
                }
            }
        }, "copy " + sourcePath + " to " + targetPath);
    }

    private boolean deleteDirectoryRecursive(FTPClient client, String parentDir, String dirName) throws IOException {
        String currentDir = client.printWorkingDirectory();
        
        try {
            client.changeWorkingDirectory(parentDir + "/" + dirName);
            
            FTPFile[] files = client.listFiles();
            if (files != null) {
                for (FTPFile file : files) {
                    if (file.getName().equals(".") || file.getName().equals("..")) {
                        continue;
                    }
                    
                    if (file.isDirectory()) {
                        deleteDirectoryRecursive(client, client.printWorkingDirectory(), file.getName());
                    } else {
                        boolean deleted = client.deleteFile(file.getName());
                        if (!deleted) {
                            log.warn("[FtpService] Failed to delete file: {}", file.getName());
                        }
                    }
                }
            }
            
            client.changeWorkingDirectory(parentDir);
            return client.removeDirectory(dirName);
            
        } finally {
            client.changeWorkingDirectory(currentDir);
        }
    }

    private void ensureDirectoryExists(FTPClient client, String path) throws IOException {
        String normalizedPath = normalizePath(path);
        if ("/".equals(normalizedPath)) {
            return;
        }
        
        String currentWorkingDir = client.printWorkingDirectory();
        if (currentWorkingDir == null) {
            currentWorkingDir = "/";
        }
        
        log.debug("[FtpService] ensureDirectoryExists - starting from: {}, target: {}", currentWorkingDir, normalizedPath);
        
        String[] pathComponents = normalizedPath.substring(1).split("/");
        String currentPath = "/";
        
        for (String component : pathComponents) {
            if (component == null || component.isEmpty()) {
                continue;
            }
            
            String nextPath = currentPath + "/" + component;
            
            if (!client.changeWorkingDirectory(nextPath)) {
                if (!client.changeWorkingDirectory(currentPath)) {
                    log.warn("[FtpService] Cannot change to parent directory: {}", currentPath);
                }
                
                boolean created = client.makeDirectory(component);
                int replyCode = client.getReplyCode();
                
                if (!created) {
                    String replyString = client.getReplyString();
                    log.error("[FtpService] Failed to create directory '{}' at '{}', reply code: {}, reply: {}", 
                        component, currentPath, replyCode, replyString);
                    
                    if (replyCode == 550) {
                        throw new FtpPermissionException("create directory", nextPath, 
                            "Permission denied: " + replyString);
                    } else if (replyCode == 553) {
                        throw new FtpPermissionException("create directory", nextPath, 
                            "Cannot create file: " + replyString);
                    } else {
                        throw new FtpException(500, "Failed to create directory: " + nextPath + 
                            ", reply code: " + replyCode + ", reply: " + replyString, String.valueOf(replyCode));
                    }
                }
                
                log.info("[FtpService] Created directory: {}", nextPath);
                
                if (!client.changeWorkingDirectory(nextPath)) {
                    log.warn("[FtpService] Created directory but cannot enter: {}", nextPath);
                }
            }
            
            currentPath = nextPath;
        }
        
        client.changeWorkingDirectory(currentWorkingDir);
    }

    private FtpFileInfo convertToFileInfo(FTPFile ftpFile, String parentDir) {
        String absolutePath = normalizePath(parentDir + "/" + ftpFile.getName());
        
        LocalDateTime lastModified = null;
        String lastModifiedStr = "";
        if (ftpFile.getTimestamp() != null) {
            lastModified = ftpFile.getTimestamp().toInstant()
                .atZone(ZoneId.systemDefault())
                .toLocalDateTime();
            lastModifiedStr = lastModified.format(DATE_FORMATTER);
        }
        
        return FtpFileInfo.builder()
            .name(ftpFile.getName())
            .path(ftpFile.getName())
            .absolutePath(absolutePath)
            .parentPath(parentDir)
            .isDirectory(ftpFile.isDirectory())
            .isFile(ftpFile.isFile())
            .isSymbolicLink(ftpFile.isSymbolicLink())
            .size(ftpFile.getSize())
            .lastModified(lastModified)
            .lastModifiedStr(lastModifiedStr)
            .permissions(ftpFile.toString().split(" ")[0])
            .owner(ftpFile.getUser())
            .group(ftpFile.getGroup())
            .hardLinkCount(ftpFile.getHardLinkCount())
            .build();
    }

    private String normalizePath(String path) {
        if (path == null || path.isEmpty()) {
            return "/";
        }
        
        String normalized = path.replace("\\", "/");
        
        while (normalized.contains("//")) {
            normalized = normalized.replace("//", "/");
        }
        
        if (!normalized.startsWith("/")) {
            normalized = "/" + normalized;
        }
        
        List<String> parts = new ArrayList<>(Arrays.asList(normalized.split("/")));
        List<String> result = new ArrayList<>();
        
        for (String part : parts) {
            if (part.isEmpty() || part.equals(".")) {
                continue;
            }
            if (part.equals("..")) {
                if (!result.isEmpty()) {
                    result.remove(result.size() - 1);
                }
            } else {
                result.add(part);
            }
        }
        
        if (result.isEmpty()) {
            return "/";
        }
        
        return "/" + String.join("/", result);
    }

    private String getParentPath(String path) {
        String normalized = normalizePath(path);
        if ("/".equals(normalized)) {
            return null;
        }
        
        int lastSlash = normalized.lastIndexOf('/');
        if (lastSlash == 0) {
            return "/";
        }
        return normalized.substring(0, lastSlash);
    }

    private String getFileName(String path) {
        String normalized = normalizePath(path);
        if ("/".equals(normalized)) {
            return "";
        }
        
        int lastSlash = normalized.lastIndexOf('/');
        return normalized.substring(lastSlash + 1);
    }

    private List<String> parsePathParts(String path) {
        String normalized = normalizePath(path);
        if ("/".equals(normalized)) {
            return Collections.singletonList("/");
        }
        
        List<String> parts = new ArrayList<>();
        parts.add("/");
        
        String[] splitParts = normalized.substring(1).split("/");
        for (String part : splitParts) {
            if (!part.isEmpty()) {
                parts.add(part);
            }
        }
        
        return parts;
    }

    private <T> T executeWithRetry(Callable<T> callable, String operation) {
        FtpProperties.Retry retryConfig = properties.getRetry();
        int maxAttempts = retryConfig.getMaxAttempts();
        long delay = retryConfig.getDelayMillis();
        double multiplier = retryConfig.getMultiplier();
        
        Exception lastException = null;
        
        for (int attempt = 1; attempt <= maxAttempts; attempt++) {
            try {
                return callable.call();
            } catch (FtpConnectionException | FtpTimeoutException e) {
                lastException = e;
                if (attempt < maxAttempts) {
                    long waitTime = (long) (delay * Math.pow(multiplier, attempt - 1));
                    log.warn("[FtpService] {} failed (attempt {}/{}), retrying in {}ms: {}", 
                        operation, attempt, maxAttempts, waitTime, e.getMessage());
                    
                    try {
                        Thread.sleep(waitTime);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        throw new FtpException("Operation interrupted: " + operation, ie);
                    }
                }
            } catch (Exception e) {
                if (e instanceof FtpException) {
                    throw (FtpException) e;
                }
                throw new FtpException("Operation failed: " + operation, e);
            }
        }
        
        if (lastException instanceof FtpException) {
            throw (FtpException) lastException;
        }
        throw new FtpException("Operation failed after " + maxAttempts + " attempts: " + operation, lastException);
    }

    public FtpFolderUploadResult uploadFolder(FtpConnectionRequest request, List<MultipartFile> files, 
            List<String> relativePaths, String targetBasePath, boolean overwrite) {
        return executeWithRetry(() -> {
            FTPClient client = null;
            String normalizedBasePath = normalizePath(targetBasePath);
            
            if (files == null || files.isEmpty()) {
                return FtpFolderUploadResult.create("", normalizedBasePath);
            }
            
            String folderName = extractFolderName(relativePaths);
            FtpFolderUploadResult result = FtpFolderUploadResult.create(folderName, normalizedBasePath);
            
            try {
                client = ftpClientPool.borrowObject(request);
                
                for (int i = 0; i < files.size(); i++) {
                    MultipartFile file = files.get(i);
                    String relativePath = (relativePaths != null && i < relativePaths.size()) 
                        ? relativePaths.get(i) : file.getOriginalFilename();
                    
                    if (relativePath == null || relativePath.isEmpty()) {
                        relativePath = file.getOriginalFilename();
                    }
                    
                    relativePath = normalizeRelativePath(relativePath);
                    String targetPath = normalizePath(normalizedBasePath + "/" + relativePath);
                    String targetDir = getParentPath(targetPath);
                    String fileName = getFileName(targetPath);
                    
                    log.info("[FtpService] Uploading folder file: {} -> {}", relativePath, targetPath);
                    
                    try {
                        ensureDirectoryExists(client, targetDir);
                        client.changeWorkingDirectory(targetDir);
                        
                        if (!overwrite) {
                            FTPFile[] existing = client.listFiles(fileName);
                            if (existing != null && existing.length > 0) {
                                result.addFailure(relativePath, targetPath, "File already exists");
                                continue;
                            }
                        }
                        
                        try (InputStream inputStream = file.getInputStream()) {
                            boolean success = client.storeFile(fileName, inputStream);
                            
                            if (!success) {
                                int replyCode = client.getReplyCode();
                                String replyString = client.getReplyString();
                                result.addFailure(relativePath, targetPath, 
                                    "Upload failed with code " + replyCode + ": " + replyString);
                            } else {
                                result.addSuccess(relativePath, targetPath, file.getSize());
                                log.info("[FtpService] Successfully uploaded folder file: {}", targetPath);
                            }
                        }
                        
                    } catch (Exception e) {
                        log.error("[FtpService] Failed to upload folder file {}: {}", relativePath, e.getMessage());
                        result.addFailure(relativePath, targetPath, e.getMessage());
                    }
                }
                
                result.setSuccess(result.isAllSuccess());
                return result;
                    
            } catch (Exception e) {
                log.error("[FtpService] Folder upload error: {}", e.getMessage());
                throw new FtpException("Folder upload error: " + e.getMessage(), e);
            } finally {
                if (client != null) {
                    ftpClientPool.returnObject(request, client);
                }
            }
        }, "upload folder to " + targetBasePath);
    }

    private String normalizeRelativePath(String path) {
        if (path == null) {
            return "";
        }
        String normalized = path.replace("\\", "/");
        while (normalized.contains("//")) {
            normalized = normalized.replace("//", "/");
        }
        while (normalized.startsWith("/")) {
            normalized = normalized.substring(1);
        }
        return normalized;
    }

    private String extractFolderName(List<String> relativePaths) {
        if (relativePaths == null || relativePaths.isEmpty()) {
            return "";
        }
        
        Set<String> firstParts = new HashSet<>();
        for (String path : relativePaths) {
            if (path == null) continue;
            String normalized = normalizeRelativePath(path);
            if (normalized.isEmpty()) continue;
            int firstSlash = normalized.indexOf('/');
            if (firstSlash > 0) {
                firstParts.add(normalized.substring(0, firstSlash));
            } else {
                firstParts.add(normalized);
            }
        }
        
        if (firstParts.size() == 1) {
            return firstParts.iterator().next();
        }
        return "";
    }

    public Map<String, Object> getPoolStats() {
        Map<String, Object> stats = new LinkedHashMap<>();
        stats.put("activeConnections", ftpClientPool.getNumActiveTotal());
        stats.put("idleConnections", ftpClientPool.getNumIdleTotal());
        stats.put("maxTotalPerKey", properties.getPool().getMaxTotal());
        stats.put("maxIdlePerKey", properties.getPool().getMaxIdle());
        stats.put("minIdlePerKey", properties.getPool().getMinIdle());
        return stats;
    }
}
