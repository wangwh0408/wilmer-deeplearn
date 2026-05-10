package com.example.demo.ftp.exception;

public class FtpTimeoutException extends FtpException {

    public FtpTimeoutException(String operation, long timeoutMs) {
        super(408, "Timeout: " + operation + " exceeded " + timeoutMs + "ms");
    }

    public FtpTimeoutException(String message) {
        super(408, "Timeout: " + message);
    }

    public FtpTimeoutException(String operation, String path, long timeoutMs) {
        super(408, "Timeout: " + operation + " on " + path + " exceeded " + timeoutMs + "ms");
    }
}
