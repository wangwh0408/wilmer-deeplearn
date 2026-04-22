package com.example.demo.controller;

import com.example.demo.common.Result;
import com.example.demo.entity.User;
import com.example.demo.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.web.bind.annotation.*;

import javax.validation.Valid;
import java.util.List;
import java.util.Optional;

@RestController
@RequestMapping("/api/users")
public class UserController {

    @Autowired
    private UserService userService;

    @GetMapping
    public Result<List<User>> list() {
        List<User> users = userService.findAll();
        return Result.success(users);
    }

    @GetMapping("/page")
    public Result<Page<User>> page(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size,
            @RequestParam(defaultValue = "id") String sort,
            @RequestParam(defaultValue = "desc") String direction) {
        
        Sort.Direction sortDirection = "asc".equalsIgnoreCase(direction) ? 
                Sort.Direction.ASC : Sort.Direction.DESC;
        Pageable pageable = PageRequest.of(page, size, Sort.by(sortDirection, sort));
        Page<User> users = userService.findAll(pageable);
        return Result.success(users);
    }

    @GetMapping("/{id}")
    public Result<User> getById(@PathVariable Long id) {
        Optional<User> user = userService.findById(id);
        return user.map(Result::success)
                .orElseGet(() -> Result.error(404, "用户不存在"));
    }

    @GetMapping("/username/{username}")
    public Result<User> getByUsername(@PathVariable String username) {
        Optional<User> user = userService.findByUsername(username);
        return user.map(Result::success)
                .orElseGet(() -> Result.error(404, "用户不存在"));
    }

    @PostMapping
    public Result<User> create(@Valid @RequestBody User user) {
        if (userService.existsByUsername(user.getUsername())) {
            return Result.error(400, "用户名已存在");
        }
        User saved = userService.save(user);
        return Result.success(201, "创建成功", saved);
    }

    @PutMapping("/{id}")
    public Result<User> update(@PathVariable Long id, @Valid @RequestBody User user) {
        Optional<User> existing = userService.findById(id);
        if (!existing.isPresent()) {
            return Result.error(404, "用户不存在");
        }
        user.setId(id);
        User updated = userService.update(user);
        return Result.success(updated);
    }

    @DeleteMapping("/{id}")
    public Result<Void> delete(@PathVariable Long id) {
        Optional<User> existing = userService.findById(id);
        if (!existing.isPresent()) {
            return Result.error(404, "用户不存在");
        }
        userService.deleteById(id);
        return Result.success();
    }
}
