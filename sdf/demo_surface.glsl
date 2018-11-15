vec3 light_source(in vec3 light_color, in vec3 position, in vec3 light_pos,
                  in vec3 normal) {
    return clamp(vec3(dot(normal, normalize(light_pos - position))) * light_color,
                 vec3(0.0), vec3(1.0));
}

vec3 surface(in vec3 origin, in vec3 position, in vec3 normal, in vec2 uv) {
    return light_source(vec3(0.5, 0.1, 0.4), position, origin,
                        normal) * 0.5 +
           light_source(vec3(0.1, 0.6, 0.3), position, vec3(0, 10, 0), normal) * 0.5;
}
