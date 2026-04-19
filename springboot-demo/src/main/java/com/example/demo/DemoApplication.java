package com.example.demo;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class DemoApplication {

    public static void main(String[] args) {
        SpringApplication.run(DemoApplication.class, args);
        System.out.println("==============================================");
        System.out.println("  Spring Boot Demo Application Started!");
        System.out.println("  Access: http://localhost:8080");
        System.out.println("  Health: http://localhost:8080/actuator/health");
        System.out.println("==============================================");
    }
}
