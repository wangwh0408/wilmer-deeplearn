package com.example.demo.ftp.exception;

import lombok.Getter;

@Getter
public class FtpException extends RuntimeException {

    private final int errorCode;
    private final String errorMessage;
    private final String ftpReplyCode;

    public FtpException(String message) {
        super(message);
        this.errorCode = 500;
        this.errorMessage = message;
        this.ftpReplyCode = null;
    }

    public FtpException(String message, Throwable cause) {
        super(message, cause);
        this.errorCode = 500;
        this.errorMessage = message;
        this.ftpReplyCode = null;
    }

    public FtpException(int errorCode, String errorMessage) {
        super(errorMessage);
        this.errorCode = errorCode;
        this.errorMessage = errorMessage;
        this.ftpReplyCode = null;
    }

    public FtpException(int errorCode, String errorMessage, String ftpReplyCode) {
        super(errorMessage + " (FTP Reply Code: " + ftpReplyCode + ")");
        this.errorCode = errorCode;
        this.errorMessage = errorMessage;
        this.ftpReplyCode = ftpReplyCode;
    }

    public FtpException(int errorCode, String errorMessage, String ftpReplyCode, Throwable cause) {
        super(errorMessage + " (FTP Reply Code: " + ftpReplyCode + ")", cause);
        this.errorCode = errorCode;
        this.errorMessage = errorMessage;
        this.ftpReplyCode = ftpReplyCode;
    }
}
