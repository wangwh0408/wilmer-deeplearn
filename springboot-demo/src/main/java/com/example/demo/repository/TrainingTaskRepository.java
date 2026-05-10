package com.example.demo.repository;

import com.example.demo.entity.TrainingTaskEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.JpaSpecificationExecutor;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface TrainingTaskRepository extends JpaRepository<TrainingTaskEntity, Long>, JpaSpecificationExecutor<TrainingTaskEntity> {

    Optional<TrainingTaskEntity> findByTaskId(String taskId);

    boolean existsByTaskId(String taskId);

    List<TrainingTaskEntity> findByStatusOrderByCreateTimeDesc(TrainingTaskEntity.Status status);

    List<TrainingTaskEntity> findByTaskTypeOrderByCreateTimeDesc(TrainingTaskEntity.TaskType taskType);

    List<TrainingTaskEntity> findAllByOrderByCreateTimeDesc();

    void deleteByTaskId(String taskId);
}
