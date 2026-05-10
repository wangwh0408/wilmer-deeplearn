package com.example.demo.ftp.exception;

public class FtpConnectionException extends FtpException {

    public FtpConnectionException(String message) {
        super(503, "FTP Connection Error: " + message);
    }

    public FtpConnectionException(String message, String ftpReplyCode) {
        super(503, "FTP Connection Error: " + message, ftpReplyCode);
    }

    public FtpConnectionException(String message, String ftpReplyCode, Throwable cause) {
        super(503, "FTP Connection Error: " + message, ftpReplyCode, cause);
    }

    public FtpConnectionException(String message, Throwable cause) {
        super(503, "FTP Connection Error: " + message, null, cause);
    }

    public FtpConnectionException(String host, int port, String message) {
        super(503, "Failed to connect to FTP server " + host + ":" + port + ": " + message);
    }

    public FtpConnectionException(String host, int port, String message, String ftpReplyCode) {
        super(503, "Failed to connect to FTP server " + host + ":" + port + ": " + message, ftpReplyCode);
    }

    public FtpConnectionException(String host, int port, String message, String ftpReplyCode, Throwable cause) {
        super(503, "Failed to connect to FTP server " + host + ":" + port + ": " + message, ftpReplyCode, cause);
    }
}
