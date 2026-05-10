package com.example.demo.ftp.pool;

import com.example.demo.ftp.config.FtpProperties;
import com.example.demo.ftp.dto.FtpConnectionRequest;
import com.example.demo.ftp.exception.FtpConnectionException;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.net.ftp.FTPClient;
import org.apache.commons.net.ftp.FTPReply;
import org.apache.commons.pool2.BaseKeyedPooledObjectFactory;
import org.apache.commons.pool2.PooledObject;
import org.apache.commons.pool2.impl.DefaultPooledObject;

import java.io.IOException;
import java.net.SocketTimeoutException;
import java.util.Objects;

@Slf4j
public class FtpClientFactory extends BaseKeyedPooledObjectFactory<FtpConnectionRequest, FTPClient> {

    private final FtpProperties properties;

    public FtpClientFactory(FtpProperties properties) {
        this.properties = properties;
    }

    @Override
    public FTPClient create(FtpConnectionRequest key) throws Exception {
        log.debug("[FtpClientFactory] Creating new FTP client for {}:{}", key.getHost(), key.getPort());
        
        FTPClient ftpClient = new FTPClient();
        
        ftpClient.setConnectTimeout(key.getConnectTimeout() > 0 ? key.getConnectTimeout() : properties.getConnectTimeout());
        ftpClient.setDefaultTimeout(key.getDefaultSoTimeout() > 0 ? key.getDefaultSoTimeout() : properties.getDefaultSoTimeout());
        ftpClient.setDataTimeout(key.getDataTimeout() > 0 ? key.getDataTimeout() : properties.getDataTimeout());
        
        ftpClient.setBufferSize(properties.getBufferSize());
        
        String encoding = key.getEncoding() != null ? key.getEncoding() : properties.getDefaultEncoding();
        ftpClient.setControlEncoding(encoding);
        
        try {
            log.debug("[FtpClientFactory] Connecting to {}:{}", key.getHost(), key.getPort());
            ftpClient.connect(key.getHost(), key.getPort());
            
            int replyCode = ftpClient.getReplyCode();
            if (!FTPReply.isPositiveCompletion(replyCode)) {
                ftpClient.disconnect();
                throw new FtpConnectionException(key.getHost(), key.getPort(), 
                    "Server refused connection with reply code: " + replyCode);
            }
            
            boolean loginSuccess = ftpClient.login(key.getUsername(), key.getPassword());
            if (!loginSuccess) {
                int loginReplyCode = ftpClient.getReplyCode();
                ftpClient.logout();
                ftpClient.disconnect();
                throw new FtpConnectionException(key.getHost(), key.getPort(),
                    "Login failed for user: " + key.getUsername(), String.valueOf(loginReplyCode));
            }
            
            if (key.isPassiveMode() || properties.isPassiveMode()) {
                log.debug("[FtpClientFactory] Enabling passive mode");
                ftpClient.enterLocalPassiveMode();
            }
            
            ftpClient.setFileType(FTPClient.BINARY_FILE_TYPE);
            ftpClient.setUseEPSVwithIPv4(false);
            
            String rootPath = key.getRootPath();
            if (rootPath != null && !rootPath.equals("/") && !rootPath.isEmpty()) {
                boolean changeDirSuccess = ftpClient.changeWorkingDirectory(rootPath);
                if (!changeDirSuccess) {
                    log.warn("[FtpClientFactory] Failed to change to root path: {}", rootPath);
                }
            }
            
            log.info("[FtpClientFactory] Successfully connected to {}:{} as {}", 
                key.getHost(), key.getPort(), key.getUsername());
            
            return ftpClient;
            
        } catch (SocketTimeoutException e) {
            throw new FtpConnectionException(key.getHost(), key.getPort(), 
                "Connection timeout after " + ftpClient.getConnectTimeout() + "ms");
        } catch (IOException e) {
            throw new FtpConnectionException(key.getHost(), key.getPort(), 
                "IO error while connecting: " + e.getMessage(), null, e);
        }
    }

    @Override
    public PooledObject<FTPClient> wrap(FTPClient ftpClient) {
        return new DefaultPooledObject<>(ftpClient);
    }

    @Override
    public void destroyObject(FtpConnectionRequest key, PooledObject<FTPClient> p) throws Exception {
        FTPClient ftpClient = p.getObject();
        if (ftpClient != null && ftpClient.isConnected()) {
            try {
                log.debug("[FtpClientFactory] Logging out and disconnecting FTP client");
                ftpClient.logout();
            } catch (Exception e) {
                log.warn("[FtpClientFactory] Error during logout: {}", e.getMessage());
            }
            try {
                ftpClient.disconnect();
            } catch (Exception e) {
                log.warn("[FtpClientFactory] Error during disconnect: {}", e.getMessage());
            }
        }
    }

    @Override
    public boolean validateObject(FtpConnectionRequest key, PooledObject<FTPClient> p) {
        FTPClient ftpClient = p.getObject();
        if (ftpClient == null || !ftpClient.isConnected()) {
            log.debug("[FtpClientFactory] Client is null or disconnected, invalid");
            return false;
        }
        
        try {
            int originalTimeout = ftpClient.getSoTimeout();
            try {
                ftpClient.setSoTimeout(5000);
                log.debug("[FtpClientFactory] Sending NOOP to validate connection");
                boolean success = ftpClient.sendNoOp();
                if (!success) {
                    log.warn("[FtpClientFactory] NOOP failed, connection may be stale");
                    return false;
                }
                
                int replyCode = ftpClient.getReplyCode();
                boolean valid = FTPReply.isPositiveCompletion(replyCode);
                if (!valid) {
                    log.warn("[FtpClientFactory] NOOP returned non-positive reply code: {}", replyCode);
                }
                return valid;
            } finally {
                ftpClient.setSoTimeout(originalTimeout);
            }
            
        } catch (java.net.SocketTimeoutException e) {
            log.warn("[FtpClientFactory] Connection validation timeout, connection may be stale");
            return false;
        } catch (Exception e) {
            log.warn("[FtpClientFactory] Error validating connection: {}", e.getMessage());
            return false;
        }
    }

    @Override
    public void activateObject(FtpConnectionRequest key, PooledObject<FTPClient> p) throws Exception {
        FTPClient ftpClient = p.getObject();
        if (ftpClient != null) {
            try {
                ftpClient.changeWorkingDirectory(key.getRootPath() != null ? key.getRootPath() : "/");
            } catch (Exception e) {
                log.warn("[FtpClientFactory] Error activating object: {}", e.getMessage());
            }
        }
    }

    @Override
    public void passivateObject(FtpConnectionRequest key, PooledObject<FTPClient> p) throws Exception {
        FTPClient ftpClient = p.getObject();
        if (ftpClient != null) {
            try {
                ftpClient.changeWorkingDirectory("/");
            } catch (Exception e) {
                log.warn("[FtpClientFactory] Error passivating object: {}", e.getMessage());
            }
        }
    }

    public static String createKey(FtpConnectionRequest request) {
        return request.getHost() + ":" + request.getPort() + ":" + request.getUsername() + ":" + request.getRootPath();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        FtpClientFactory that = (FtpClientFactory) o;
        return Objects.equals(properties, that.properties);
    }

    @Override
    public int hashCode() {
        return Objects.hash(properties);
    }
}
