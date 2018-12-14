extern crate glium;
extern crate nalgebra_glm as glm;

use glium::glutin;

const MOUSE_SPEED: f32 = 0.04;
const MOVE_SPEED: f32 = 0.01;


pub struct CameraState {
    aspect_ratio: f32,
    position: glm::Vec3,
    direction: glm::Vec3,

    moving_up: bool,
    moving_left: bool,
    moving_down: bool,
    moving_right: bool,
    moving_forward: bool,
    moving_backward: bool,

    cursor_pos: Option<glm::Vec2>,
}

impl CameraState {
    pub fn new() -> CameraState {
        CameraState {
            aspect_ratio: 1024.0 / 768.0,
            position: glm::vec3(0.1, 0.1, 1.0),
            direction: glm::vec3(0.0, 0.0, -1.0),
            moving_up: false,
            moving_left: false,
            moving_down: false,
            moving_right: false,
            moving_forward: false,
            moving_backward: false,
            cursor_pos: None,
        }
    }

    pub fn set_position(&mut self, pos: (f32, f32, f32)) {
        self.position = glm::vec3(pos.0, pos.1, pos.2);
    }

    pub fn set_direction(&mut self, dir: (f32, f32, f32)) {
        self.direction = glm::vec3(dir.0, dir.1, dir.2);
    }

    pub fn get_perspective(&self) -> glm::Mat4 {
        let fov: f32 = 3.141592 / 2.0;
        let zfar = 1024.0;
        let znear = 0.1;

        let f = 1.0 / (fov / 2.0).tan();

        // note: remember that this is column-major, so the lines of code are actually columns
        glm::Mat4::new(
            f / self.aspect_ratio,
            0.0,
            0.0,
            0.0,
            0.0,
            f,
            0.0,
            0.0,
            0.0,
            0.0,
            (zfar + znear) / (zfar - znear),
            1.0,
            0.0,
            0.0,
            -(2.0 * zfar * znear) / (zfar - znear),
            0.0,
        )
        .transpose()
    }

    pub fn get_view(&self) -> glm::Mat4 {
        let f = {
            let f = self.direction;
            let len = f.x * f.x + f.y * f.y + f.z * f.z;
            let len = len.sqrt();
            glm::vec3(f.x / len, f.y / len, f.z / len)
        };

        let up = glm::vec3(0.0, 1.0, 0.0);

        let s = glm::vec3(
            f.y * up.z - f.z * up.y,
            f.z * up.x - f.x * up.z,
            f.x * up.y - f.y * up.x,
        )
        .normalize();

        let u = glm::vec3(
            s.y * f.z - s.z * f.y,
            s.z * f.x - s.x * f.z,
            s.x * f.y - s.y * f.x,
        );

        let p = glm::vec3(
            -self.position.x * s.x - self.position.y * s.y - self.position.z * s.z,
            -self.position.x * u.x - self.position.y * u.y - self.position.z * u.z,
            -self.position.x * f.x - self.position.y * f.y - self.position.z * f.z,
        );

        // note: remember that this is column-major, so the lines of code are actually columns
        glm::Mat4::new(
            s.x, u.x, f.x, 0.0, s.y, u.y, f.y, 0.0, s.z, u.z, f.z, 0.0, p.x, p.y, p.z, 1.0,
        )
        .transpose()
    }

    pub fn update(&mut self) {
        let f = {
            let f = self.direction;
            let len = f.x * f.x + f.y * f.y + f.z * f.z;
            let len = len.sqrt();
            glm::vec3(f.x / len, f.y / len, f.z / len)
        };

        let up = glm::vec3(0.0, 1.0, 0.0);

        let s = glm::vec3(
            f.y * up.z - f.z * up.y,
            f.z * up.x - f.x * up.z,
            f.x * up.y - f.y * up.x,
        )
        .normalize();

        let u = glm::vec3(
            s.y * f.z - s.z * f.y,
            s.z * f.x - s.x * f.z,
            s.x * f.y - s.y * f.x,
        );

        if self.moving_up {
            self.position.x += u.x * MOVE_SPEED;
            self.position.y += u.y * MOVE_SPEED;
            self.position.z += u.z * MOVE_SPEED;
        }

        if self.moving_left {
            self.position.x -= s.x * MOVE_SPEED;
            self.position.y -= s.y * MOVE_SPEED;
            self.position.z -= s.z * MOVE_SPEED;
        }

        if self.moving_down {
            self.position.x -= u.x * MOVE_SPEED;
            self.position.y -= u.y * MOVE_SPEED;
            self.position.z -= u.z * MOVE_SPEED;
        }

        if self.moving_right {
            self.position.x += s.x * MOVE_SPEED;
            self.position.y += s.y * MOVE_SPEED;
            self.position.z += s.z * MOVE_SPEED;
        }

        if self.moving_forward {
            self.position.x += f.x * MOVE_SPEED;
            self.position.y += f.y * MOVE_SPEED;
            self.position.z += f.z * MOVE_SPEED;
        }

        if self.moving_backward {
            self.position.x -= f.x * MOVE_SPEED;
            self.position.y -= f.y * MOVE_SPEED;
            self.position.z -= f.z * MOVE_SPEED;
        }
    }

    fn update_camera_rotation(&mut self, diff: &glm::Vec2) {
        let angle = glm::vec2(
            f32::atan2(self.direction.z, self.direction.x),
            f32::atan2(
                (self.direction.x * self.direction.x + self.direction.z * self.direction.z).sqrt(),
                self.direction.y,
            ),
        );
        let angle = angle + diff;
        let angle = glm::vec2(
            angle.x,
            f32::min(std::f32::consts::PI - 0.1, f32::max(angle.y, 0.1)),
        );

        self.direction = glm::vec3(
            angle.y.sin() * angle.x.cos(),
            angle.y.cos(),
            angle.y.sin() * angle.x.sin(),
        );
    }

    pub fn reset_camera(&mut self) {
        self.cursor_pos = None;
    }

    pub fn process_input(&mut self, event: &glutin::WindowEvent, dt: u64, handle_camera: bool) {
        match *event {
            glutin::WindowEvent::KeyboardInput { input, .. } => {
                let pressed = input.state == glutin::ElementState::Pressed;
                let key = match input.virtual_keycode {
                    Some(key) => key,
                    None => return,
                };
                match key {
                    glutin::VirtualKeyCode::Z => self.moving_up = pressed,
                    glutin::VirtualKeyCode::X => self.moving_down = pressed,
                    glutin::VirtualKeyCode::A => self.moving_left = pressed,
                    glutin::VirtualKeyCode::D => self.moving_right = pressed,
                    glutin::VirtualKeyCode::W => self.moving_forward = pressed,
                    glutin::VirtualKeyCode::S => self.moving_backward = pressed,
                    glutin::VirtualKeyCode::Left => {
                        self.update_camera_rotation(&glm::vec2(-MOUSE_SPEED * dt as f32, 0.0))
                    }
                    glutin::VirtualKeyCode::Right => {
                        self.update_camera_rotation(&glm::vec2(MOUSE_SPEED * dt as f32, 0.0))
                    }
                    glutin::VirtualKeyCode::Up => {
                        self.update_camera_rotation(&glm::vec2(0.0, -MOUSE_SPEED * dt as f32))
                    }
                    glutin::VirtualKeyCode::Down => {
                        self.update_camera_rotation(&glm::vec2(0.0, MOUSE_SPEED * dt as f32))
                    }
                    _ => (),
                };
            }
            glutin::WindowEvent::CursorMoved { position, .. } => {
                if handle_camera {
                    let new_pos = glm::vec2(position.x as f32, position.y as f32);
                    let diff = match self.cursor_pos {
                        None => glm::vec2(0.0, 0.0),
                        Some(old_pos) => new_pos - old_pos,
                    };

                    let diff = diff * (MOUSE_SPEED * dt as f32);
                    self.update_camera_rotation(&diff);

                    self.cursor_pos = Some(new_pos);
                }
            }
            _ => return,
        }
    }
}
