
#define M_PI 3.1415926535897932384626433832795

vec2 uv(in vec3 p) {
    float theta = (atan(p.y, p.x) + M_PI) / (2 * M_PI);
    float phi = (atan(length(p.xy), p.z) + M_PI) / (2 * M_PI);
    return vec2(theta, phi);
}
