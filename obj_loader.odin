package main

import "core:strings"
import "core:strconv"
import "core:mem"
import "core:os"
import "core:fmt"
import "core:slice"
import "core:runtime"

Obj :: struct {
    allocator: mem.Allocator,
    vertices: [dynamic]Vec3,
    colors: [dynamic]Vec3,
    texture_coords: [dynamic]Vec2,
    indices: [dynamic]u32,
}

load_object :: proc(file: string, file_allocator := context.allocator, obj_allocator := context.allocator) -> (loaded: Obj, ok: bool) {
    loaded.allocator = obj_allocator
    loaded.vertices = make([dynamic]Vec3, obj_allocator)
    loaded.texture_coords = make([dynamic]Vec2, obj_allocator)
    loaded.indices = make([dynamic]u32, obj_allocator)

    v_offset, t_offset: int
    obj_file := os.read_entire_file(MODEL_PATH, file_allocator) or_return
    defer delete(obj_file, file_allocator)

    str_obj_file := string(obj_file)
    for line in strings.split_lines_iterator(&str_obj_file) {
        if len(line) == 0 do continue

        switch line[0] {
        case 'v':
            switch line[1] {
            case ' ':
                line := line[2:]
                vertices := strings.split(line, " ", context.temp_allocator)
                x, x_ok := strconv.parse_f32(strings.trim_space(vertices[0]))
                y, y_ok := strconv.parse_f32(strings.trim_space(vertices[1]))
                z, z_ok := strconv.parse_f32(strings.trim_space(vertices[2]))
                if !(x_ok && y_ok && z_ok) {
                    fmt.println("not ok -", vertices)
                    assert(false)
                }
                append(&loaded.vertices, Vec3{x, y, z})
            case 't':
                line := line[3:]
                textures := strings.split(line, " ", context.temp_allocator)
                u, u_ok := strconv.parse_f32(strings.trim_space(textures[0]))
                v, v_ok := strconv.parse_f32(strings.trim_space(textures[1]))
                assert(u_ok && v_ok)
                append(&loaded.texture_coords, Vec2{u, 1.0 - v})
            }
        case 'f':
            line := line[2:]
            faces := strings.split(line, " ", context.temp_allocator)
            x, y, z := strings.split(faces[0], "/", context.temp_allocator), strings.split(faces[1], "/", context.temp_allocator), strings.split(faces[2], "/", context.temp_allocator)
            x_vert, xv_ok := strconv.parse_int(strings.trim_space(x[0]))
            x_text, xt_ok := strconv.parse_int(strings.trim_space(x[1]))
            y_vert, yv_ok := strconv.parse_int(strings.trim_space(y[0]))
            y_text, yt_ok := strconv.parse_int(strings.trim_space(y[1]))
            z_vert, zv_ok := strconv.parse_int(strings.trim_space(z[0]))
            z_text, zt_ok := strconv.parse_int(strings.trim_space(z[1]))

            assert(xv_ok && xt_ok && yv_ok && yt_ok && zv_ok && zt_ok)
            assert(x_vert == x_text && y_vert == y_text && z_vert == z_text, "some of the coords are not the same")
            
            append(&loaded.indices, u32(x_vert - 1), u32(y_vert - 1), u32(z_vert - 1))
        }
    }

    loaded.colors = make([dynamic]Vec3, len(loaded.vertices), obj_allocator)
    slice.fill(loaded.colors[:], Vec3{1.0, 1.0, 1.0})

    ok = true

    return
}

load_object_unique_vertices :: proc(file: string, file_allocator := context.allocator, obj_allocator := context.allocator, temp_allocator := context.temp_allocator) -> (loaded: Obj, ok: bool) {
    obj := load_object(file, temp_allocator, temp_allocator) or_return

    loaded.allocator = obj_allocator
    loaded.vertices = make([dynamic]Vec3, obj_allocator)
    loaded.texture_coords = make([dynamic]Vec2, obj_allocator)
    loaded.indices = make([dynamic]u32, obj_allocator)
    loaded.colors = make([dynamic]Vec3, obj_allocator)

    Unique_Vertex :: [8]f32

    unique_map := make(map[Unique_Vertex]int, 1<<runtime.MAP_MIN_LOG2_CAPACITY, temp_allocator)
    s := soa_zip(v = obj.vertices[:], t = obj.texture_coords[:], c = obj.colors[:])
    for v, i in s {
        uv := Unique_Vertex{v.v.x, v.v.y, v.v.z, v.t.x, v.t.y, v.c.x, v.c.y, v.c.z}
        index, exists := unique_map[uv]
        if !exists {
            new_index := len(loaded.vertices)
            unique_map[uv] = new_index
            append(&loaded.vertices, v.v)
            append(&loaded.texture_coords, v.t)
            append(&loaded.colors, v.c)
        }
    }

    for index in obj.indices {
        entry := s[index]
        v, t, c := expand_values(entry)
        uv := Unique_Vertex{v.x, v.y, v.z, t.x, t.y, c.x, c.y, c.z}
        new_index := unique_map[uv]
        append(&loaded.indices, u32(new_index))
    }

    ok = true
    return
}

unload_object :: proc(obj: Obj) {
    delete(obj.vertices)
    delete(obj.texture_coords)
    delete(obj.indices)
    delete(obj.colors)
}