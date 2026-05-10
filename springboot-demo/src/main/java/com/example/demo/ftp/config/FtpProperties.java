package com.example.demo.ftp.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.Map;

@Data
@Component
@ConfigurationProperties(prefix = "ftp")
public class FtpProperties {

    private Map<String, FtpServerConfig> servers = new HashMap<>();

    private Pool pool = new Pool();

    private Retry retry = new Retry();

    private int bufferSize = 8192;

    private int connectTimeout = 30000;

    private int dataTimeout = 300000;

    private int defaultSoTimeout = 60000;

    private boolean passiveMode = true;

    private String defaultEncoding = "UTF-8";

    private boolean autoDetectEncoding = true;

    @Data
    public static class FtpServerConfig {

        private String host;

        private int port = 21;

        private String username = "anonymous";

        private String password = "";

        private String rootPath = "/";

        private int connectTimeout = 30000;

        private int dataTimeout = 300000;

        private int defaultSoTimeout = 60000;

        private boolean passiveMode = true;

        private String encoding = "UTF-8";
    }

    @Data
    public static class Pool {

        private int maxTotal = 20;

        private int maxIdle = 10;

        private int minIdle = 2;

        private int maxWaitMillis = 30000;

        private boolean testOnBorrow = true;

        private boolean testOnReturn = true;

        private boolean testWhileIdle = true;

        private long timeBetweenEvictionRunsMillis = 120000;

        private long minEvictableIdleTimeMillis = 600000;

        private long softMinEvictableIdleTimeMillis = 300000;

        private int numTestsPerEvictionRun = 5;

        private boolean testOnCreate = false;
    }

    @Data
    public static class Retry {

        private int maxAttempts = 3;

        private long delayMillis = 1000;

        private double multiplier = 2.0;
    }
}
