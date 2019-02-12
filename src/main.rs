extern crate nalgebra_glm as glm;

extern crate notify;

extern crate termion;
extern crate tui;

extern crate chrono;

#[macro_use]
extern crate glium;

use notify::{DebouncedEvent, RecommendedWatcher, RecursiveMode, Watcher};

use glium::glutin::{self, Event, WindowEvent};
use glium::index::PrimitiveType;
use glium::Surface;

use chrono::{TimeZone, Utc};
use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};
use std::env::args;
use std::fs;
use std::io::{Read, Write};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

mod camera;
mod export;

const FRAME_TIME_BUFFER_SIZE: usize = 30;

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

const GRID_SDF: &'static str = "
uniform sampler3D grid_sdf;
uniform float scale_factor;

float sdf(in vec3 p) {
    return texture(grid_sdf, p / scale_factor).r;
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
    let sdf_shader = if args().len() < 2 {
        shaders["sdf_shader"].clone()
    } else {
        GRID_SDF.to_owned()
    };
    format!(
        "
    #version 330

    uniform mat4 inv;
    uniform vec3 origin;
    uniform float time;
    uniform float visualization_depth;
    uniform bool enable_visualization;

    uniform float near;

    in vec2 ndc;
    out vec4 color;

    #define EPS 1e-3

    {noise_common}

    {noise2D}

    {noise3D}

    {hg_source}

    {sdf_source}

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

    // Returns color, updates ray_origin to final position
    bool trace_ray(inout vec3 ray_origin, in vec3 ray_vector) {{
        float radius = 0;
        float total_traveled = 0;

        vec3 current_point = ray_origin;

        // Perform a few iterations of sphere tracing,
        // exit if we're too far away
        for (int k = 0; k < 60; k++) {{
            radius = sdf(current_point);
            total_traveled += radius;
            current_point += ray_vector * radius;

            if (total_traveled > 100.0) {{
                return false;
            }}
        }}

        ray_origin = current_point;

        return true;
    }}

    {uv_source}

    {surface_source}

    float visualize_sdf(in vec3 ray_origin, in vec3 ray_vec) {{
        float accum = 0.0;
        int num_steps = 30;
        vec3 current_point = ray_origin;

        for (int i = 0; i < num_steps; i++) {{
            float radius = sdf(current_point);
            current_point += ray_vec * visualization_depth;
            accum += radius;
        }}

        accum = clamp(accum, 0.0, float(num_steps) * 100.0);
        return mod(accum / (float(num_steps) * 100.0), 1.0);
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

        if (enable_visualization) {{
            color = vec4(visualize_sdf(current_point, ray_vec));
        }} else {{
            if (trace_ray(current_point, ray_vec)) {{
                vec2 uv_val = uv(current_point);
                vec3 normal_sample_pt = current_point - ray_vec * EPS;
                vec3 normal = sobel_gradient_estimate(normal_sample_pt);
                color = vec4(surface(origin, current_point, normal, uv_val), 0);
            }} else {{
                color = vec4(0.0);
            }}
        }}
    }}
",
        noise_common = shaders["noise_common"],
        noise2D = shaders["noise2D"],
        noise3D = shaders["noise3D"],
        hg_source = shaders["hg_shader"],
        sdf_source = sdf_shader,
        uv_source = shaders["uv_shader"],
        surface_source = shaders["surface_shader"],
    )
}

fn compile<'a>(
    display: &glium::Display,
    fs_shader: &'a str,
) -> Result<glium::Program, glium::ProgramCreationError> {
    glium::Program::from_source(display, PASSTHROUGH_VS, fs_shader, None)
}

fn update_shader(
    display: &glium::Display,
    current_shader_name: &String,
    shaders: &mut Shaders,
    recv_channel: &mpsc::Receiver<String>,
    app: &mut TerminalApp,
) -> Option<glium::Program> {
    match recv_channel.try_recv() {
        Ok(sdf_string) => {
            if sdf_string != shaders[current_shader_name] {
                shaders.insert(current_shader_name.clone(), sdf_string);
                preprocess_shaders(shaders);
                match compile(&display, &generate_sdf_shader(shaders)) {
                    Ok(program) => {
                        if app.anchors == (None, None) {
                            app.right_pane = "No errors reported".to_owned();
                        }
                        app.alert = (app.alert.0, false);
                        Some(program)
                    }
                    Err(msg) => {
                        app.right_pane = format!("{}", msg);
                        app.alert = (app.alert.0, true);
                        None
                    }
                }
            } else {
                None
            }
        }
        Err(_) => None,
    }
}

struct TerminalApp {
    left_pane: String,
    right_pane: String,
    alert: (bool, bool),
    size: tui::layout::Rect,
    anchors: (Option<glm::Vec3>, Option<glm::Vec3>),
}

impl Default for TerminalApp {
    fn default() -> TerminalApp {
        TerminalApp {
            left_pane: String::new(),
            right_pane: String::new(),
            alert: (false, false),
            size: tui::layout::Rect::default(),
            anchors: (None, None),
        }
    }
}

type TermionTerminal = tui::Terminal<
    tui::backend::TermionBackend<
        termion::screen::AlternateScreen<
            termion::input::MouseTerminal<termion::raw::RawTerminal<std::io::Stdout>>,
        >,
    >,
>;

fn terminal_ui_init() -> TermionTerminal {
    use termion::raw::IntoRawMode;

    let stdout = std::io::stdout().into_raw_mode().unwrap();
    let stdout = termion::input::MouseTerminal::from(stdout);
    let stdout = termion::screen::AlternateScreen::from(stdout);
    let backend = tui::backend::TermionBackend::new(stdout);
    let mut terminal = tui::Terminal::new(backend).unwrap();
    terminal.hide_cursor().unwrap();

    terminal
}

fn terminal_ui_resize(app: &mut TerminalApp, terminal: &mut TermionTerminal) {
    let size = terminal.size().unwrap();
    if size != app.size {
        terminal.resize(size).unwrap();
        app.size = size;
    }
}

fn terminal_ui_draw(app: &mut TerminalApp, terminal: &mut TermionTerminal) {
    use tui::layout::{Alignment, Constraint, Direction, Layout};
    use tui::style::{Color, Modifier, Style};
    use tui::widgets::{Block, Borders, Paragraph, Text, Widget};

    let size = terminal.size().unwrap();

    terminal
        .draw(|mut f| {
            tui::widgets::Block::default()
                .style(tui::style::Style::default().bg(tui::style::Color::DarkGray))
                .render(&mut f, size);
            let chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(40), Constraint::Percentage(40)].as_ref())
                .split(size);
            let left_text = if app.alert.0 {
                Text::styled(
                    &app.left_pane,
                    Style::default().fg(Color::Red).modifier(Modifier::Bold),
                )
            } else {
                Text::styled(&app.left_pane, Style::default().fg(Color::Green))
            };

            let right_text = if app.alert.1 {
                Text::styled(
                    &app.right_pane,
                    Style::default().fg(Color::Red).modifier(Modifier::Bold),
                )
            } else {
                Text::styled(
                    &app.right_pane,
                    Style::default().fg(Color::White).modifier(Modifier::Italic),
                )
            };

            let mut left_block = Block::default().borders(Borders::ALL);
            left_block.render(&mut f, chunks[0]);
            Paragraph::new(vec![left_text].iter())
                .block(left_block)
                .alignment(Alignment::Left)
                .wrap(true)
                .render(&mut f, chunks[0]);
            let mut right_block = Block::default().borders(Borders::ALL);
            right_block.render(&mut f, chunks[1]);
            Paragraph::new(vec![right_text].iter())
                .block(right_block)
                .alignment(Alignment::Left)
                .wrap(true)
                .render(&mut f, chunks[1]);
        })
        .unwrap();
}

fn main() {
    let mut events_loop = glutin::EventsLoop::new();
    let window = glutin::WindowBuilder::new();
    let context = glutin::ContextBuilder::new();
    let display = glium::Display::new(window, context, &events_loop).unwrap();

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
        )
        .unwrap()
    };

    let index_buffer =
        glium::IndexBuffer::new(&display, PrimitiveType::TriangleStrip, &[1 as u16, 2, 0, 3])
            .unwrap();

    let mut shaders: Shaders = HashMap::new();
    shaders.insert(
        "noise_common".to_owned(),
        fs::read_to_string("sdf/noise_common.glsl").unwrap(),
    );
    shaders.insert(
        "noise2D".to_owned(),
        fs::read_to_string("sdf/noise2D.glsl").unwrap(),
    );
    shaders.insert(
        "noise3D".to_owned(),
        fs::read_to_string("sdf/noise3D.glsl").unwrap(),
    );
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

    let mut term = terminal_ui_init();
    let mut term_app = TerminalApp::default();

    let mut camera = camera::CameraState::new();
    camera.set_position((0.0, 0.0, 10.0));
    camera.set_direction((0.0, 0.0, -1.0));

    let mut accumulator = Duration::new(0, 0);
    let mut previous_clock = Instant::now();
    let start_clock = previous_clock.clone();

    let mut frame_time_buffer: VecDeque<u64> = VecDeque::new();

    let gl_window = display.gl_window();
    let window = gl_window.window();

    let mut inner_size;
    let mut physical_inner_size;

    //let mut virtual_resolution = (inner_size.width as u32, inner_size.height as u32);
    let mut virtual_resolution = (500, 500);
    let mut grabbed: bool = false;
    let mut visualization_depth: f32 = 0.2;
    let mut enable_visualization: bool = false;

    let grid_texture = if args().len() >= 2 {
        let args_vec: Vec<String> = args().collect();
        let grid_sdf = export::grid_sdf_read(&args_vec[1]);

        let raw = glium::texture::RawImage3d {
            data: Cow::from(&grid_sdf),
            width: export::GRID_SDF_DIM as u32,
            height: export::GRID_SDF_DIM as u32,
            depth: export::GRID_SDF_DIM as u32,
            format: glium::texture::ClientFormat::F32,
        };
        Some(
            glium::texture::Texture3d::with_mipmaps(
                &display,
                raw,
                glium::texture::MipmapsOption::NoMipmap,
            )
            .unwrap(),
        )
    } else {
        None
    };

    // drawing a frame
    loop {
        // TODO: only need to rebuild the display texture after resize!
        let display_texture = glium::texture::Texture2d::empty_with_format(
            &display,
            glium::texture::UncompressedFloatFormat::U8U8U8U8,
            glium::texture::MipmapsOption::NoMipmap,
            virtual_resolution.0,
            virtual_resolution.1,
        )
        .unwrap();
        let mut framebuffer =
            glium::framebuffer::SimpleFrameBuffer::new(&display, &display_texture).unwrap();

        // Update first person perspective
        camera.update();
        inner_size = window.get_inner_size().unwrap();
        physical_inner_size = inner_size.to_physical(window.get_hidpi_factor());

        let mut target = display.draw();

        // Get new views from camera
        let inv: glm::Mat4 = glm::inverse(&(camera.get_perspective() * camera.get_view()));
        let origin: glm::Vec3 =
            (glm::inverse(&camera.get_view()) * glm::vec4(0.0, 0.0, 0.0, 1.0)).xyz();

        // Render SDF
        target.clear_color(0.0, 1.0, 0.0, 1.0);
        let elapsed = ((Instant::now() - start_clock).as_secs() as f32)
            + ((Instant::now() - start_clock).subsec_micros() as f32 / 1_000_000.0);

        match &grid_texture {
            Some(grid_texture) => {
                framebuffer
                    .draw(
                        &vertex_buffer,
                        &index_buffer,
                        &program,
                        &uniform! {
                            inv: to_mat4(inv),
                            origin: to_vec3(origin),
                            near: 0.1 as f32,
                            time: elapsed,
                            visualization_depth: visualization_depth,
                            enable_visualization: enable_visualization,
                            grid_sdf: grid_texture.sampled()
                                .magnify_filter(
                                    glium::uniforms::MagnifySamplerFilter::Linear)
                                .wrap_function(
                                    glium::uniforms::SamplerWrapFunction::Clamp),
                            scale_factor: 0.1f32,
                        },
                        &Default::default(),
                    )
                    .unwrap();
            }
            None => {
                framebuffer
                    .draw(
                        &vertex_buffer,
                        &index_buffer,
                        &program,
                        &uniform! {
                            inv: to_mat4(inv),
                            origin: to_vec3(origin),
                            near: 0.1 as f32,
                            time: elapsed,
                            visualization_depth: visualization_depth,
                            enable_visualization: enable_visualization,
                        },
                        &Default::default(),
                    )
                    .unwrap();
            }
        }
        target.blit_from_simple_framebuffer(
            &framebuffer,
            &glium::Rect {
                left: 0,
                bottom: 0,
                width: virtual_resolution.0,
                height: virtual_resolution.1,
            },
            &glium::BlitTarget {
                left: 0,
                bottom: 0,
                width: physical_inner_size.width as i32,
                height: physical_inner_size.height as i32,
            },
            glium::uniforms::MagnifySamplerFilter::Nearest,
        );
        target.finish().unwrap();

        // VBlank
        let now = Instant::now();
        accumulator += now - previous_clock;

        let current_frame_time = accumulator.subsec_millis() as u64 + accumulator.as_secs() * 1000;
        frame_time_buffer.push_back(current_frame_time);
        if frame_time_buffer.len() > FRAME_TIME_BUFFER_SIZE {
            frame_time_buffer.pop_front();
        }
        let frame_time =
            frame_time_buffer.iter().fold(0, |a, b| a + b) as f64 / frame_time_buffer.len() as f64;
        if frame_time_buffer.len() == FRAME_TIME_BUFFER_SIZE {
            term_app.alert = (true, term_app.alert.1);

            frame_time_buffer.clear();
        } else {
            term_app.alert = (false, term_app.alert.1);
        }

        term_app.left_pane = format!(
            "FPS: {}\nframe_time: {}\nVRes: {:?}\nPRes: {:?}",
            1000.0 / frame_time,
            frame_time,
            virtual_resolution,
            (physical_inner_size.width, physical_inner_size.height)
        );

        previous_clock = now;

        let fixed_time_stamp = Duration::new(0, 16666667);
        while accumulator >= fixed_time_stamp {
            accumulator -= fixed_time_stamp;
        }

        // Handle terminal resize and update
        terminal_ui_resize(&mut term_app, &mut term);
        terminal_ui_draw(&mut term_app, &mut term);

        // Handle events
        let mut should_exit = false;
        events_loop.poll_events(|event| match event {
            Event::WindowEvent { event, window_id } => {
                if window_id == display.gl_window().id() {
                    match event {
                        WindowEvent::CloseRequested => should_exit = true,
                        WindowEvent::MouseInput {
                            button: glutin::MouseButton::Left,
                            state: glutin::ElementState::Pressed,
                            ..
                        } => {
                            window.grab_cursor(true).ok();
                            window.hide_cursor(true);
                            grabbed = true;
                        }
                        WindowEvent::MouseInput {
                            button: glutin::MouseButton::Left,
                            state: glutin::ElementState::Released,
                            ..
                        } => {
                            window.grab_cursor(false).ok();
                            window.hide_cursor(false);
                            grabbed = false;
                            camera.reset_camera();
                        }
                        WindowEvent::KeyboardInput {
                            input:
                                glutin::KeyboardInput {
                                    virtual_keycode: Some(glutin::VirtualKeyCode::Space),
                                    state: glutin::ElementState::Released,
                                    ..
                                },
                            ..
                        } => {
                            let pos = camera.get_position();
                            let (anchors, repr) = match term_app.anchors {
                                (None, None) | (Some(_), Some(_)) => {
                                    ((Some(pos), None), Some(format!("Anchor 1: {})", pos)))
                                }
                                (Some(first_anchor), None) => (
                                    (Some(first_anchor), Some(pos)),
                                    Some(format!("Anchor 1: {}\nAnchor 2: {}", first_anchor, pos)),
                                ),
                                // Shouldn't happen?
                                _ => ((Some(pos), None), None),
                            };
                            term_app.anchors = anchors;

                            match repr {
                                Some(repr) => term_app.right_pane = repr,
                                None => {}
                            };
                        }
                        WindowEvent::KeyboardInput {
                            input:
                                glutin::KeyboardInput {
                                    virtual_keycode: Some(glutin::VirtualKeyCode::Return),
                                    state: glutin::ElementState::Released,
                                    ..
                                },
                            ..
                        } => match term_app.anchors.clone() {
                            (Some(a1), Some(a2)) => {
                                let grid_sdf = export::grid_sdf_async_compute(
                                    &display,
                                    &shaders["sdf_shader"],
                                    elapsed,
                                    (a1, a2),
                                );

                                let timestamp = Utc::now().timestamp();
                                let filename = format!("grid_sdf_{}.sdf", timestamp);
                                export::grid_sdf_write(&filename, &grid_sdf);

                                term_app.anchors = (None, None);

                                term_app.right_pane =
                                    format!("Saved grid SDF to file {}", filename);
                                term_app.alert.1 = false;
                            }
                            _ => {
                                term_app.right_pane =
                                    "Need both anchors set to compute grid SDF".to_owned();
                                term_app.alert.1 = true;
                            }
                        },
                        WindowEvent::KeyboardInput {
                            input:
                                glutin::KeyboardInput {
                                    virtual_keycode: Some(glutin::VirtualKeyCode::Comma),
                                    ..
                                },
                            ..
                        } => {
                            virtual_resolution.0 = u32::max(virtual_resolution.0 - 100, 100);
                            virtual_resolution.1 = u32::max(virtual_resolution.1 - 100, 100);
                        }
                        WindowEvent::KeyboardInput {
                            input:
                                glutin::KeyboardInput {
                                    virtual_keycode: Some(glutin::VirtualKeyCode::Period),
                                    ..
                                },
                            ..
                        } => {
                            virtual_resolution.0 =
                                u32::min(virtual_resolution.0 + 100, inner_size.width as u32);
                            virtual_resolution.1 =
                                u32::min(virtual_resolution.1 + 100, inner_size.height as u32);
                        }
                        WindowEvent::KeyboardInput {
                            input:
                                glutin::KeyboardInput {
                                    virtual_keycode: Some(glutin::VirtualKeyCode::V),
                                    state: glutin::ElementState::Released,
                                    ..
                                },
                            ..
                        } => {
                            enable_visualization = !enable_visualization;
                        }
                        WindowEvent::KeyboardInput {
                            input:
                                glutin::KeyboardInput {
                                    virtual_keycode: Some(glutin::VirtualKeyCode::LBracket),
                                    ..
                                },
                            ..
                        } => {
                            visualization_depth -= 0.03;
                        }
                        WindowEvent::KeyboardInput {
                            input:
                                glutin::KeyboardInput {
                                    virtual_keycode: Some(glutin::VirtualKeyCode::RBracket),
                                    ..
                                },
                            ..
                        } => {
                            visualization_depth += 0.03;
                        }
                        ev => {
                            camera.process_input(&ev, current_frame_time as u64, grabbed);
                        }
                    }
                }
            }
            _ => (),
        });

        // See if anything has changed
        if let Some(prog) = update_shader(
            &display,
            &"sdf_shader".to_owned(),
            &mut shaders,
            &watchers.sdf_update,
            &mut term_app,
        ) {
            program = prog;
        }

        if let Some(prog) = update_shader(
            &display,
            &"uv_shader".to_owned(),
            &mut shaders,
            &watchers.uv_update,
            &mut term_app,
        ) {
            program = prog;
        }

        if let Some(prog) = update_shader(
            &display,
            &"surface_shader".to_owned(),
            &mut shaders,
            &watchers.surface_update,
            &mut term_app,
        ) {
            program = prog;
        }

        // Exit if the user clicks the X
        if should_exit {
            return;
        }
    }
}
