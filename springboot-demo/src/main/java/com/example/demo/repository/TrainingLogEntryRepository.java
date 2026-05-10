package com.example.demo.repository;

import com.example.demo.entity.TrainingLogEntryEntity;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.JpaSpecificationExecutor;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface TrainingLogEntryRepository extends JpaRepository<TrainingLogEntryEntity, Long>, JpaSpecificationExecutor<TrainingLogEntryEntity> {

    List<TrainingLogEntryEntity> findByTaskIdOrderBySequenceAsc(String taskId);

    List<TrainingLogEntryEntity> findByTaskIdAndSequenceGreaterThanOrderBySequenceAsc(String taskId, Integer sequence);

    Page<TrainingLogEntryEntity> findByTaskIdOrderBySequenceAsc(String taskId, Pageable pageable);

    List<TrainingLogEntryEntity> findByTaskIdOrderByIdAsc(String taskId);

    @Query("SELECT MAX(e.sequence) FROM TrainingLogEntryEntity e WHERE e.taskId = :taskId")
    Integer findMaxSequenceByTaskId(@Param("taskId") String taskId);

    @Query("SELECT COUNT(e) FROM TrainingLogEntryEntity e WHERE e.taskId = :taskId")
    long countByTaskId(@Param("taskId") String taskId);

    void deleteByTaskId(String taskId);

    List<TrainingLogEntryEntity> findByTaskIdAndLevelOrderBySequenceAsc(String taskId, String level);
}
