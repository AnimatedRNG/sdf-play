#[allow(unused_imports)]
#[macro_use]
extern crate conrod;

extern crate nalgebra_glm as glm;

extern crate notify;

use notify::{DebouncedEvent, RecommendedWatcher, RecursiveMode, Watcher};

use conrod::backend::glium::glium;

use glium::glutin::{self, Event, WindowEvent};
use glium::index::PrimitiveType;
use glium::Surface;
use glium::{implement_vertex, uniform};

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

mod camera;

const PASSTHROUGH_VS: &'static str = "
#version 330
in vec2 position;
in vec2 tex_coords;
out vec2 ndc;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    ndc = tex_coords;
}
";

const SAMPLE_SDF: &'static str = "
float sdf(in vec3 p) {
    vec2 q = vec2(length(p.xz) - 3.0, p.y);
    return length(q) - 2.0;
}
";

const SAMPLE_UV: &'static str = "
#define M_PI 3.1415926535897932384626433832795

vec2 uv(in vec3 p) {
    float theta = (atan(p.y, p.x) + M_PI) / (2 * M_PI);
    float phi = (atan(length(p.xy), p.z) + M_PI) / (2 * M_PI);
    return vec2(theta, phi);
}
";

const SAMPLE_SURFACE: &'static str = "
vec3 light_source(in vec3 light_color, in vec3 position, in vec3 light_pos,
                  in vec3 normal) {
    return clamp(vec3(dot(normal, normalize(light_pos - position))) * light_color,
                 vec3(0.0), vec3(1.0));
}

vec3 surface(in vec3 origin, in vec3 position, in vec3 normal, in vec2 uv) {
    return light_source(vec3(1.0, 1.0, 1.0), position, origin,
                        normal);
}
";

fn to_mat4(mat: glm::Mat4) -> [[f32; 4]; 4] {
    [
        [mat[(0, 0)], mat[(1, 0)], mat[(2, 0)], mat[(3, 0)]],
        [mat[(0, 1)], mat[(1, 1)], mat[(2, 1)], mat[(3, 1)]],
        [mat[(0, 2)], mat[(1, 2)], mat[(2, 2)], mat[(3, 2)]],
        [mat[(0, 3)], mat[(1, 3)], mat[(2, 3)], mat[(3, 3)]],
    ]
}

fn to_vec3(mat: glm::Vec3) -> [f32; 3] {
    [mat[0], mat[1], mat[2]]
}

#[allow(dead_code)]
struct Watchers {
    sdf_update: mpsc::Receiver<String>,
    uv_update: mpsc::Receiver<String>,
    surface_update: mpsc::Receiver<String>,

    sdf_watcher: RecommendedWatcher,
    uv_watcher: RecommendedWatcher,
    surface_watcher: RecommendedWatcher,
}

type Shaders = HashMap<String, String>;

fn file_update_thread<'a>(
    file_name: &'a str,
    rx: mpsc::Receiver<DebouncedEvent>,
    tx: mpsc::Sender<String>,
) {
    loop {
        match rx.recv() {
            Ok(_) => match fs::read_to_string(file_name) {
                Ok(file_contents) => {
                    tx.send(file_contents).unwrap();
                }
                Err(_) => {
                    println!("Did you delete {}? I can't read it", file_name);
                }
            },
            Err(e) => println!("watch error: {:?}", e),
        }
    }
}

fn read_handle<'a, 'b>(file_name: &'a str, default_value: &'b str) -> String {
    match fs::File::open(file_name) {
        Ok(mut f_) => {
            let mut contents = String::new();
            f_.read_to_string(&mut contents).unwrap();

            contents
        }
        Err(_) => {
            let mut f_ = fs::File::create(file_name).unwrap();
            f_.write(default_value.as_bytes()).unwrap();
            default_value.to_string()
        }
    }
}

fn init_watchers() -> Watchers {
    let sdf_filename = "sdf/root.glsl";
    let parameterization_filename = "sdf/uv.glsl";
    let surface_filename = "sdf/surface.glsl";

    let sdf_data = read_handle(sdf_filename, SAMPLE_SDF);
    let parameterization_data = read_handle(parameterization_filename, SAMPLE_UV);
    let surface_data = read_handle(surface_filename, SAMPLE_SURFACE);

    // TODO: Refactor this mess
    let (sdf_update_channel_tx, sdf_update_channel_rx) = mpsc::channel();
    let (uv_update_channel_tx, uv_update_channel_rx) = mpsc::channel();
    let (surface_update_channel_tx, surface_update_channel_rx) = mpsc::channel();

    let (sdf_content_update_channel_tx, sdf_content_update_channel_rx) = mpsc::channel();
    let (uv_content_update_channel_tx, uv_content_update_channel_rx) = mpsc::channel();
    let (surface_content_update_channel_tx, surface_content_update_channel_rx) = mpsc::channel();

    let mut sdf_watcher: RecommendedWatcher =
        Watcher::new(sdf_update_channel_tx, Duration::from_secs(1)).unwrap();
    sdf_watcher
        .watch(sdf_filename, RecursiveMode::NonRecursive)
        .unwrap();

    let mut uv_watcher: RecommendedWatcher =
        Watcher::new(uv_update_channel_tx, Duration::from_secs(1)).unwrap();
    uv_watcher
        .watch(parameterization_filename, RecursiveMode::NonRecursive)
        .unwrap();

    let mut surface_watcher: RecommendedWatcher =
        Watcher::new(surface_update_channel_tx, Duration::from_secs(1)).unwrap();
    surface_watcher
        .watch(surface_filename, RecursiveMode::NonRecursive)
        .unwrap();

    sdf_content_update_channel_tx.send(sdf_data).unwrap();
    uv_content_update_channel_tx
        .send(parameterization_data)
        .unwrap();
    surface_content_update_channel_tx
        .send(surface_data)
        .unwrap();

    thread::spawn(move || {
        file_update_thread(
            sdf_filename,
            sdf_update_channel_rx,
            sdf_content_update_channel_tx.clone(),
        );
    });

    thread::spawn(move || {
        file_update_thread(
            parameterization_filename,
            uv_update_channel_rx,
            uv_content_update_channel_tx.clone(),
        );
    });

    thread::spawn(move || {
        file_update_thread(
            surface_filename,
            surface_update_channel_rx,
            surface_content_update_channel_tx.clone(),
        );
    });

    Watchers {
        sdf_update: sdf_content_update_channel_rx,
        uv_update: uv_content_update_channel_rx,
        surface_update: surface_content_update_channel_rx,

        sdf_watcher: sdf_watcher,
        uv_watcher: uv_watcher,
        surface_watcher: surface_watcher,
    }
}

fn preprocess_shaders(shaders: &mut Shaders) {
    shaders
        .iter_mut()
        .for_each(|(_, v)| *v = format!("#line 0\n{}", v))
}

fn generate_sdf_shader<'a>(shaders: &Shaders) -> String {
    format!(
        "
    #version 330

    uniform mat4 inv;
    uniform vec3 origin;

    uniform float near;

    in vec2 ndc;
    out vec4 color;

    #define EPS 1e-3

    {hg_source}

    {sdf_source}

    {uv_source}

    {surface_source}

    float h(in vec3 p, in uint index) {{
        vec3 forward = p;
        vec3 backward = p;
        forward[index] += EPS;
        backward[index] -= EPS;
        return dot(vec3(sdf(backward), sdf(p), sdf(forward)), vec3(1.0, 2.0, 1.0));
    }}

    float h_p(in vec3 p, in uint index) {{
        vec3 forward = p;
        vec3 backward = p;
        forward[index] += EPS;
        backward[index] -= EPS;
        return dot(vec2(sdf(backward), sdf(forward)), vec2(1.0, -1.0));
    }}

    vec3 sobel_gradient_estimate(in vec3 p) {{
        float h_x = h_p(p, uint(0)) * h(p, uint(1)) * h(p, uint(2));
        float h_y = h_p(p, uint(1)) * h(p, uint(2)) * h(p, uint(0));
        float h_z = h_p(p, uint(2)) * h(p, uint(0)) * h(p, uint(1));

        return normalize(-vec3(h_x, h_y, h_z));
    }}

    vec3 simple_gradient_estimate(in vec3 p) {{
        float h_x = sdf(p + vec3(EPS, 0.0, 0.0)) - sdf(p - vec3(EPS, 0.0, 0.0));
        float h_y = sdf(p + vec3(0.0, EPS, 0.0)) - sdf(p - vec3(0.0, EPS, 0.0));
        float h_z = sdf(p + vec3(0.0, 0.0, EPS)) - sdf(p - vec3(0.0, 0.0, EPS));

        return normalize(vec3(h_x, h_y, h_z));
    }}

    void main() {{
        // Compute the ray vector
        vec4 clip_space = vec4(ndc * 2.0 - 1.0, 1.0, 1.0);
        vec4 tmp = inv * clip_space;
        tmp /= tmp.w;
        tmp -= vec4(origin, 0);
        vec3 ray_vec = normalize(vec3(tmp));

        // This is where our ray is currently at
        vec3 current_point = origin + ray_vec * near;
        float radius = 0;
        float total_traveled = 0;

        // Perform a few iterations of sphere tracing,
        // exit if we're too far away
        for (int k = 0; k < 25; k++) {{
            radius = sdf(current_point);
            total_traveled += radius;
            current_point += ray_vec * radius;

            if (total_traveled > 100.0) {{
                color = vec4(0);
                return;
            }}
        }}

        // Really simple lighting model
        vec2 uv_val = uv(current_point);
        vec3 normal_sample_pt = current_point - ray_vec * EPS * 100.0;
        vec3 normal = sobel_gradient_estimate(normal_sample_pt);
        color = vec4(surface(origin, current_point, normal, uv_val), 0);
    }}
",
        hg_source = shaders["hg_shader"],
        sdf_source = shaders["sdf_shader"],
        uv_source = shaders["uv_shader"],
        surface_source = shaders["surface_shader"],
    )
}

fn compile<'a>(display: &glium::Display, fs_shader: &'a str) -> Option<glium::Program> {
    match glium::Program::from_source(display, PASSTHROUGH_VS, fs_shader, None) {
        Ok(prog) => Some(prog),
        Err(err_msg) => {
            println!("{}", err_msg);
            None
        }
    }
}

fn update_shader(
    display: &glium::Display,
    current_shader_name: &String,
    shaders: &mut Shaders,
    recv_channel: &mpsc::Receiver<String>,
) -> Option<glium::Program> {
    match recv_channel.try_recv() {
        Ok(sdf_string) => {
            if sdf_string != shaders[current_shader_name] {
                shaders.insert(current_shader_name.clone(), sdf_string);
                preprocess_shaders(shaders);
                compile(&display, &generate_sdf_shader(shaders))
            } else {
                None
            }
        }
        Err(_) => None,
    }
}

fn main() {
    let mut events_loop = glutin::EventsLoop::new();
    let window = glutin::WindowBuilder::new();
    let context = glutin::ContextBuilder::new();
    let display = glium::Display::new(window, context, &events_loop).unwrap();

    let mut ui = conrod::UiBuilder::new([400 as f64, 400 as f64]).build();
    let image_map = conrod::image::Map::<glium::texture::Texture2d>::new();
    let mut renderer = conrod::backend::glium::Renderer::new(&display).unwrap();

    let watchers = init_watchers();

    let vertex_buffer = {
        #[derive(Copy, Clone)]
        struct Vertex {
            position: [f32; 2],
            tex_coords: [f32; 2],
        }

        implement_vertex!(Vertex, position, tex_coords);

        glium::VertexBuffer::new(
            &display,
            &[
                Vertex {
                    position: [-1.0, -1.0],
                    tex_coords: [0.0, 0.0],
                },
                Vertex {
                    position: [-1.0, 1.0],
                    tex_coords: [0.0, 1.0],
                },
                Vertex {
                    position: [1.0, 1.0],
                    tex_coords: [1.0, 1.0],
                },
                Vertex {
                    position: [1.0, -1.0],
                    tex_coords: [1.0, 0.0],
                },
            ],
        ).unwrap()
    };

    let index_buffer =
        glium::IndexBuffer::new(&display, PrimitiveType::TriangleStrip, &[1 as u16, 2, 0, 3])
            .unwrap();

    let mut shaders: Shaders = HashMap::new();
    shaders.insert(
        "hg_shader".to_owned(),
        fs::read_to_string("sdf/hg_sdf.glsl").unwrap(),
    );
    shaders.insert("sdf_shader".to_owned(), watchers.sdf_update.recv().unwrap());
    shaders.insert("uv_shader".to_owned(), watchers.uv_update.recv().unwrap());
    shaders.insert(
        "surface_shader".to_owned(),
        watchers.surface_update.recv().unwrap(),
    );

    let mut program = compile(&display, &generate_sdf_shader(&shaders)).unwrap();

    let mut camera = camera::CameraState::new();
    camera.set_position((0.0, 0.0, 10.0));
    camera.set_direction((0.0, 0.0, -1.0));

    let mut accumulator = Duration::new(0, 0);
    let mut previous_clock = Instant::now();

    // drawing a frame
    loop {
        // Update first person perspective
        camera.update();

        let mut target = display.draw();

        // Get new views from camera
        let inv: glm::Mat4 = glm::inverse(&(camera.get_perspective() * camera.get_view()));
        let origin: glm::Vec3 =
            (glm::inverse(&camera.get_view()) * glm::vec4(0.0, 0.0, 0.0, 1.0)).xyz();

        // Render SDF
        target.clear_color(0.0, 1.0, 0.0, 1.0);
        target
            .draw(
                &vertex_buffer,
                &index_buffer,
                &program,
                &uniform! {
                    inv: to_mat4(inv),
                    origin: to_vec3(origin),
                    near: 0.1 as f32,
                },
                &Default::default(),
            ).unwrap();

        if let Some(primitives) = ui.draw_if_changed() {
            renderer.fill(&display, primitives, &image_map);
            renderer.draw(&display, &mut target, &image_map).unwrap();
        }
        target.finish().unwrap();

        // Handle events
        let mut should_exit = false;
        events_loop.poll_events(|event| match event {
            Event::WindowEvent { event, window_id } => if window_id == display.gl_window().id() {
                match event {
                    WindowEvent::Closed => should_exit = true,
                    ev => camera.process_input(&ev),
                }
            },
            _ => (),
        });

        // See if anything has changed
        if let Some(prog) = update_shader(
            &display,
            &"sdf_shader".to_owned(),
            &mut shaders,
            &watchers.sdf_update,
        ) {
            program = prog;
        }

        if let Some(prog) = update_shader(
            &display,
            &"uv_shader".to_owned(),
            &mut shaders,
            &watchers.uv_update,
        ) {
            program = prog;
        }

        if let Some(prog) = update_shader(
            &display,
            &"surface_shader".to_owned(),
            &mut shaders,
            &watchers.surface_update,
        ) {
            program = prog;
        }

        // Exit if the user clicks the X
        if should_exit {
            return;
        }

        // VBlank
        let now = Instant::now();
        accumulator += now - previous_clock;
        previous_clock = now;

        let fixed_time_stamp = Duration::new(0, 16666667);
        while accumulator >= fixed_time_stamp {
            accumulator -= fixed_time_stamp;
        }
    }
}
