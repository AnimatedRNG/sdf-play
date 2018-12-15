extern crate byteorder;
extern crate glium;

use self::byteorder::{ByteOrder, LittleEndian};
use std::fs;
use std::io::{BufReader, BufWriter, Read, Write};

const GRID_SDF_DIM: usize = 32;
const GRID_SDF_SIZE: usize = GRID_SDF_DIM * GRID_SDF_DIM * GRID_SDF_DIM;
const GRID_SDF_ELEM_SIZE: usize = 4;

pub type GridSDF = [[[f32; GRID_SDF_DIM]; GRID_SDF_DIM]; GRID_SDF_DIM];

pub fn grid_sdf_async_compute(
    display: &glium::backend::Facade,
    sdf: &str,
    time: f32,
    anchors: (glm::Vec3, glm::Vec3),
) -> GridSDF {
    struct GridSDF {
        data: [f32],
    }

    implement_buffer_content!(GridSDF);
    implement_uniform_block!(GridSDF, data);

    let program = glium::program::ComputeShader::from_source(
        display,
        &format!(
            "\
            #version 430
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            #define GRID_SDF_DIM {GRID_SDF_DIM}

            uniform float time;
            uniform vec3 anchor_0;
            uniform vec3 anchor_1;

            layout(std430) buffer sdf_block {{
                float data[];
            }};

            {sdf}

            void main() {{
                uint index = gl_GlobalInvocationID.x * GRID_SDF_DIM * GRID_SDF_DIM +
                    gl_GlobalInvocationID.y * GRID_SDF_DIM + gl_GlobalInvocationID.z;
                vec3 span = anchor_1 - anchor_0;
                vec3 interp = vec3(gl_GlobalInvocationID.xyz) / float(GRID_SDF_DIM);
                vec3 pos = anchor_0 + span * interp;
                data[index] = sdf(pos);
            }}
",
            GRID_SDF_DIM = GRID_SDF_DIM,
            sdf = sdf
        ),
    )
    .unwrap();

    let mut buffer = glium::uniforms::UniformBuffer::<GridSDF>::empty_unsized(
        display,
        GRID_SDF_SIZE * GRID_SDF_ELEM_SIZE,
    )
    .unwrap();

    /*{
        let mut mapping = buffer.map();
        for val in mapping.data.iter_mut() {
            *val = -1.0;
        }
    }*/

    program.execute(
        uniform!( sdf_block: &*buffer,
                  time: time,
                  anchor_0: (anchors.0.x, anchors.0.y, anchors.0.z),
                  anchor_1: (anchors.1.x, anchors.1.y, anchors.1.z) ),
        GRID_SDF_DIM as u32,
        GRID_SDF_DIM as u32,
        GRID_SDF_DIM as u32,
    );

    {
        let mapping = buffer.map();
        let mut sdf_data_3d = [[[0.0; GRID_SDF_DIM]; GRID_SDF_DIM]; GRID_SDF_DIM];

        // This is awful :/
        for i in 0..GRID_SDF_DIM {
            for j in 0..GRID_SDF_DIM {
                for k in 0..GRID_SDF_DIM {
                    let index = i * GRID_SDF_DIM * GRID_SDF_DIM + j * GRID_SDF_DIM + k;
                    sdf_data_3d[i][j][k] = mapping.data[index];
                }
            }
        }
        sdf_data_3d
    }
}

pub fn grid_sdf_read(filename: &str) -> GridSDF {
    let mut grid_sdf = [[[0.0; GRID_SDF_DIM]; GRID_SDF_DIM]; GRID_SDF_DIM];
    let mut buffer = Vec::new();

    let mut file = BufReader::new(
        fs::File::open(filename).expect(&format!("Unable to read grid SDF from {}", filename)),
    );

    let buffer_data_len = file.read_to_end(&mut buffer).unwrap();
    assert!(buffer_data_len == 4 * GRID_SDF_DIM * GRID_SDF_DIM * GRID_SDF_DIM + 4 * 3);

    let mut ptr: usize = 0;

    {
        let mut read_u32 = || {
            let result = LittleEndian::read_u32(&mut buffer[ptr..ptr + 4]);
            ptr += 4;
            return result;
        };

        assert!(read_u32() == GRID_SDF_DIM as u32);
        assert!(read_u32() == GRID_SDF_DIM as u32);
        assert!(read_u32() == GRID_SDF_DIM as u32);
    }

    {
        let mut read_f32 = || {
            let result = LittleEndian::read_f32(&mut buffer[ptr..ptr + 4]);
            ptr += 4;
            return result;
        };

        for i in 0..GRID_SDF_DIM {
            for j in 0..GRID_SDF_DIM {
                for k in 0..GRID_SDF_DIM {
                    grid_sdf[i][j][k] = read_f32();
                }
            }
        }
    }

    grid_sdf
}

pub fn grid_sdf_write(filename: &str, grid: &GridSDF) {
    let mut file = BufWriter::new(
        fs::File::create(filename).expect(&format!("Unable to write grid SDF to {}", filename)),
    );

    {
        let mut write_u32 = |data: usize| {
            let mut buf = [0; 4];
            LittleEndian::write_u32(&mut buf, data as u32);
            file.write(&buf).unwrap();
        };

        write_u32(GRID_SDF_DIM);
        write_u32(GRID_SDF_DIM);
        write_u32(GRID_SDF_DIM);
    }

    for i in 0..GRID_SDF_DIM {
        for j in 0..GRID_SDF_DIM {
            for k in 0..GRID_SDF_DIM {
                let mut buf: [u8; 4] = [0; 4];

                LittleEndian::write_f32(&mut buf, grid[i][j][k]);

                file.write(&buf).unwrap();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_write() {
        let grid_sdf = [[[-1.0; GRID_SDF_DIM]; GRID_SDF_DIM]; GRID_SDF_DIM];

        grid_sdf_write("/tmp/test_sdf", &grid_sdf);
        let read = grid_sdf_read("/tmp/test_sdf");

        assert!(grid_sdf == read);
    }
}
