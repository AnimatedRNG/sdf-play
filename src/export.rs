extern crate byteorder;
extern crate glium;
extern crate nalgebra_glm as glm;

use self::byteorder::{ByteOrder, LittleEndian};
use std::fs;
use std::io::{BufReader, BufWriter, Read, Write};

use ndarray::prelude::*;

pub const GRID_SDF_DIM: usize = 32;
pub const GRID_SDF_SIZE: usize = GRID_SDF_DIM * GRID_SDF_DIM * GRID_SDF_DIM;
pub const GRID_SDF_ELEM_SIZE: usize = 4;

//pub type GridSDF = Vec<f32>;
//#[derive(Serialize, Deserialize)]

#[derive(h5::H5Type, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct GridSDFMetadata {
    resolution_x: usize,
    resolution_y: usize,
    resolution_z: usize,

    start_x: f32,
    start_y: f32,
    start_z: f32,

    end_x: f32,
    end_y: f32,
    end_z: f32,
}

pub struct GridSDF {
    pub metadata: GridSDFMetadata,

    pub data: Array1<f32>,
}

pub fn grid_sdf_async_compute(
    display: &glium::backend::Facade,
    sdf: &str,
    time: f32,
    anchors: (glm::Vec3, glm::Vec3),
) -> GridSDF {
    struct GPUGridSDF {
        data: [f32],
    }

    implement_buffer_content!(GPUGridSDF);
    implement_uniform_block!(GPUGridSDF, data);

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

    let mut buffer = glium::uniforms::UniformBuffer::<GPUGridSDF>::empty_unsized(
        display,
        GRID_SDF_SIZE * GRID_SDF_ELEM_SIZE,
    )
    .unwrap();

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

        GridSDF {
            metadata: GridSDFMetadata {
                resolution_x: GRID_SDF_DIM,
                resolution_y: GRID_SDF_DIM,
                resolution_z: GRID_SDF_DIM,

                start_x: anchors.0.x,
                start_y: anchors.0.y,
                start_z: anchors.0.z,

                end_x: anchors.1.x,
                end_y: anchors.1.y,
                end_z: anchors.1.z,
            },
            data: Array::from_vec(mapping.data.to_vec()),
        }
    }
}

pub fn grid_sdf_read(filename: &str) -> GridSDF {
    let file = h5::File::open(filename, "r").unwrap();

    let metadata = file.dataset("metadata").unwrap();

    let metadata = metadata.read_1d::<GridSDFMetadata>().unwrap();

    let metadata = metadata[0].clone();

    let data = file.dataset("data").unwrap();
    let data = data.read_2d::<f32>().unwrap();
    let data: Array1<_> = data.row(0).to_owned();

    GridSDF {
        metadata: metadata,
        data: data,
    }
}

pub fn grid_sdf_write(filename: &str, grid: &GridSDF) {
    let file = h5::File::open(filename, "w").unwrap();

    let metadata = file
        .new_dataset::<GridSDFMetadata>()
        .create("metadata", 1)
        .unwrap();
    metadata.write(&[grid.metadata.clone()]).unwrap();

    let data = file
        .new_dataset::<f32>()
        .create("data", (1, GRID_SDF_SIZE))
        .unwrap();
    data.write(
        &grid
            .data
            .clone()
            .into_shape(IxDyn(&[1, grid.data.shape()[0]]))
            .unwrap(),
    )
    .unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_write() {
        let grid_len = GRID_SDF_DIM * GRID_SDF_DIM * GRID_SDF_DIM;
        let grid_sdf: Vec<_> = (0..grid_len).map(|i| i as f32).collect();
        let grid_sdf = GridSDF {
            metadata: GridSDFMetadata {
                resolution_x: 32,
                resolution_y: 32,
                resolution_z: 32,

                start_x: 0f32,
                start_y: 0f32,
                start_z: 0f32,

                end_x: 1f32,
                end_y: 1f32,
                end_z: 1f32,
            },
            data: arr1(&grid_sdf),
        };

        grid_sdf_write("/tmp/test_sdf.hdf5", &grid_sdf);
        let read = grid_sdf_read("/tmp/test_sdf.hdf5");

        assert!(grid_sdf.data == read.data);
    }
}
