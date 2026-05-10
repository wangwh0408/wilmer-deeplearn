package com.example.demo.ftp.config;

import com.example.demo.ftp.pool.FtpClientFactory;
import com.example.demo.ftp.pool.FtpClientPool;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class FtpPoolConfig {

    @Bean
    public FtpClientFactory ftpClientFactory(FtpProperties properties) {
        return new FtpClientFactory(properties);
    }

    @Bean
    public FtpClientPool ftpClientPool(FtpClientFactory factory, FtpProperties properties) {
        return new FtpClientPool(factory, properties);
    }
}
