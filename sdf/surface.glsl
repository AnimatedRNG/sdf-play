vec3 light_source(in vec3 light_color, in vec3 position, in vec3 light_pos,
                  in vec3 normal) {
    vec3 light_vec = normalize(light_pos - position);
    vec3 pos = position + light_vec * 2.0;
    if (!trace_ray(pos, light_vec)) {
        return clamp(vec3(dot(normal, light_vec)) * light_color,
                     vec3(0.0), vec3(1.0));
    } else {
        return vec3(0.0, 0.0, 0.0);
    }
}

vec3 surface(in vec3 origin, in vec3 position, in vec3 normal, in vec2 uv) {
    return light_source(vec3(0.9, 0.9, 0.9), position, vec3(10.0, 30.0, 0.0),
                        normal) +
        light_source(vec3(0.1, 0.1, 0.1), position, origin,
                        normal);
}
