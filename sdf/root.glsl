float torus(in vec3 p) {
    vec2 q = vec2(length(p.xz) - 3.0, p.y);
    return length(q) - 2.0;
}

float sdf(in vec3 p) {
    pMod3(p, vec3(35.0));
    float shape1 = torus(p);
    float shape2 = length(p - 8.0 * vec3(cos(time), sin(time), 0.0)) - 3.0;
    return min(shape1, shape2);
    return shape2;
}
